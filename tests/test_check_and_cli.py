"""Tests for AST discovery, probe check Gates 1-3, lockfile caching, and all CLI commands."""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import probe
from probe import ProbeHaltError
from probe.backends import BackendError, CheckScore
from probe.check import _load_manual_fixture, _sha256_text, _validate_gate3_snapshot, discover_probes, run_probe_check
from probe.cli import app
from probe.config import (
    CheckConfig,
    DecisionConfig,
    ModelEndpointConfig,
    ProbeConfig,
    RuntimeConfig,
    TextConfig,
)
from probe.runtime import ProbeType, resolve_module_name
from probe.store import ProbeStore


def _write_sample_project(root: Path) -> Path:
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "sample-app"
            version = "0.1.0"

            [tool.probe.models.decision]
            provider = "http_decision"
            url = "http://mock.local/decision"
            model = "local-decision"

            [tool.probe.models.text]
            provider = "openai_compat"
            url = "http://mock.local/v1"
            model = "qwen3.5:9b"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    pkg = root / "src" / "tutor"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    mod = pkg / "turn.py"
    mod.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe(
                [
                    "Tutor opens with a short Spanish greeting",
                    "Tutor does NOT give away the target Spanish sentence",
                ],
                id="tutor.opening_turn",
            )
            def generate_opening_turn(lesson: str) -> str:
                probe.here(
                    "Draft stays grounded in lesson vocabulary",
                    id="tutor.draft",
                    kind=probe.type.text,
                    lesson=lesson,
                    draft="Hola",
                )
                return "Hola! Try greeting the barista in Spanish."
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return mod


def test_discover_and_gates_with_caching_and_fixtures(tmp_path: Path) -> None:
    _write_sample_project(tmp_path)
    discovered = discover_probes(root=tmp_path)
    assert [(d.probe_id, d.site, d.probe_type.value) for d in discovered] == [
        ("tutor.opening_turn", "decorator", "decision"),
        ("tutor.draft", "here", "text"),
    ]

    # Add a manual fixture for tutor.draft so Gate 3 tests both fixture and synthetic paths.
    fixtures_dir = tmp_path / ".probes" / "fixtures"
    fixtures_dir.mkdir(parents=True)
    (fixtures_dir / "tutor.draft.json").write_text(
        json.dumps(
            {
                "pass": {"watched": {"lesson": "coffee", "draft": "GOOD grounded"}},
                "fail": {"watched": {"lesson": "coffee", "draft": "BAD ungrounded"}},
            }
        ),
        encoding="utf-8",
    )

    http_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        http_calls["n"] += 1
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if url.endswith("/chat/completions"):
            sys_msg = body["messages"][0]["content"]
            if "audit @probe and probe.here" in sys_msg:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "Checks inspect Spanish greeting/teaching and grounding in watched variables."
                                    )
                                }
                            }
                        ]
                    },
                )
            if "passing_snapshot" in sys_msg:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "passing_snapshot": {
                                                "inputs": {"lesson": "coffee"},
                                                "output": "GOOD Hola, say you want coffee",
                                            },
                                            "failing_snapshot": {
                                                "inputs": {"lesson": "coffee"},
                                                "output": "BAD Quiero un cafe",
                                            },
                                        }
                                    )
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": f"Analysis of {body['messages'][1]['content'][:80]}"}}]},
            )
        if url.endswith("/decision"):
            if body["probe_id"].endswith("#gate2"):
                return httpx.Response(200, json={"probabilities": [0.95 for _ in body["checks"]]})
            snap = body["snapshot"]
            ok = "GOOD" in snap
            probs = [0.95 if ok else 0.10 for _ in body["checks"]]
            return httpx.Response(200, json={"probabilities": probs})
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        text=TextConfig(),
        check=CheckConfig(),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    ok, results = run_probe_check(config=cfg, transport=httpx.MockTransport(handler))
    assert ok is True
    assert [r.status for r in results] == ["passed", "passed"]
    first_calls = http_calls["n"]
    assert first_calls > 0
    assert (tmp_path / ".probes" / "lock.json").is_file()

    # Second run skips Gates 2-3 cleanly via cache (zero HTTP calls).
    ok2, results2 = run_probe_check(config=cfg, transport=httpx.MockTransport(handler))
    assert ok2 is True
    assert [r.status for r in results2] == ["cached", "cached"]
    assert http_calls["n"] == first_calls

    # --force re-runs Gates 2-3.
    ok3, results3 = run_probe_check(force=True, config=cfg, transport=httpx.MockTransport(handler))
    assert ok3 is True
    assert [r.status for r in results3] == ["passed", "passed"]
    assert http_calls["n"] > first_calls


def test_gate1_and_gate2_rejections(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe(["x > 0", "y > 0", "z > 0"], id="dup.id")
            def fn_a(x: int, y: int, z: int) -> int:
                return x

            @probe("still dup", id="dup.id")
            def fn_b() -> int:
                return 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen"),
        check=CheckConfig(max_checks=2, max_check_chars=200),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok, results = run_probe_check(config=cfg)
    assert ok is False
    assert all(r.status == "failed" and r.gate == "gate1" for r in results)

    # Gate 2 rejection on deterministic math check.
    (tmp_path / "bad.py").write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Sum equals a + b", id="math.add")
            def add(a: int, b: int) -> int:
                return a + b
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    def reject_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Check [0] tests exact Python arithmetic a + b; write a Python assert instead."
                                )
                            }
                        }
                    ]
                },
            )
        if url.endswith("/decision"):
            return httpx.Response(200, json={"probabilities": [0.05, 0.95, 0.90]})
        raise AssertionError(f"Unexpected URL: {url}")

    ok_g2, res_g2 = run_probe_check(config=cfg, transport=httpx.MockTransport(reject_handler))
    assert ok_g2 is False
    assert len(res_g2) == 1
    assert res_g2[0].gate == "gate2"
    assert "assert" in res_g2[0].reason.lower()


def test_cli_list_report_log_runs_clean(tmp_path: Path, monkeypatch) -> None:
    _write_sample_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    runs_dir = tmp_path / ".probes" / "runs"
    store = ProbeStore(runs_dir, create=True)
    try:
        # Older run + newer run with out-of-order insertion to verify call_seq ASC ordering.
        store.create_run(
            "01RUNOLD000000000000000001",
            started_at="2026-11-18T10:00:00Z",
            cmd="pytest old",
            pid=101,
            decision_model="http_decision:local-decision",
            text_model="openai_compat:qwen3.5:9b",
        )
        store.finish_run("01RUNOLD000000000000000001", status="completed", finished_at="2026-11-18T10:00:05Z")

        run_id = "01RUNNEW000000000000000002"
        store.create_run(
            run_id,
            started_at="2026-11-18T15:10:04Z",
            cmd="pytest evals/test_tutor_turns.py",
            pid=202,
            decision_model="http_decision:local-decision",
            text_model="openai_compat:qwen3.5:9b",
        )
        # Insert call #2 before call #1 and call #3 to verify call_seq ASC sorting.
        store.insert_event(
            run_id=run_id,
            call_seq=2,
            elapsed_ms=412.0,
            started_at="2026-11-18T15:10:04.412Z",
            probe_id="tutor.opening_turn",
            site="decorator",
            probe_type="decision",
            file_line="src/tutor/turn.py:3",
            checks=("Opens Spanish", "No spoiler"),
            payload_json='{"output": "Hola, dime algo en español."}',
            check_scores=(CheckScore("Opens Spanish", 0.94, True), CheckScore("No spoiler", 0.91, True)),
            verdict="pass",
        )
        store.insert_event(
            run_id=run_id,
            call_seq=1,
            elapsed_ms=88.0,
            started_at="2026-11-18T15:10:04.088Z",
            probe_id="rag.draft_answer",
            site="here",
            probe_type="text",
            file_line="src/rag/synth.py:94",
            checks=("Grounded",),
            payload_json='{"watched": {"draft": "Chunk #1"}}',
            check_scores=(CheckScore("Grounded", 0.96, True),),
            verdict="pass",
            text_analysis="Cites only filing 2026-11-16.",
        )
        store.insert_event(
            run_id=run_id,
            call_seq=3,
            elapsed_ms=785.0,
            started_at="2026-11-18T15:10:04.785Z",
            probe_id="tutor.opening_turn",
            site="decorator",
            probe_type="decision",
            file_line="src/tutor/turn.py:3",
            checks=("Opens Spanish", "No spoiler"),
            payload_json='{"output": "Try saying: Quiero un café"}',
            check_scores=(CheckScore("Opens Spanish", 0.97, True), CheckScore("No spoiler", 0.06, False)),
            verdict="fail",
            text_analysis="Tutor gives away target Spanish phrase verbatim.",
        )
        store.finish_run(run_id, status="halted", halt_call_seq=3, finished_at="2026-11-18T15:10:05Z")
    finally:
        store.close()

    runner = CliRunner()

    list_res = runner.invoke(app, ["list", "--json"])
    assert list_res.exit_code == 0
    list_data = json.loads(list_res.stdout)
    assert [p["probe_id"] for p in list_data["probes"]] == ["tutor.opening_turn", "tutor.draft"]
    assert all(p["lock_status"] == "unlocked" for p in list_data["probes"])

    rep_default = runner.invoke(app, ["report"])
    assert rep_default.exit_code == 0
    assert "HALTED on FAIL at call #003" in rep_default.stdout
    assert "#003" in rep_default.stdout
    assert "#001" not in rep_default.stdout  # passed hidden by default

    rep_all = runner.invoke(app, ["report", "--all"])
    assert rep_all.exit_code == 0
    pos1 = rep_all.stdout.index("  #001  ")
    pos2 = rep_all.stdout.index("  #002  ")
    pos3 = rep_all.stdout.index("  #003  ")
    assert pos1 < pos2 < pos3

    rep_md = runner.invoke(app, ["report", "--all", "--format", "md"])
    assert rep_md.exit_code == 0
    assert "## Run" in rep_md.stdout

    log_multi = runner.invoke(app, ["log", "tutor.opening_turn", "rag.*", "--payload"])
    assert log_multi.exit_code == 0
    lpos1 = log_multi.stdout.index("#001")
    lpos2 = log_multi.stdout.index("#002")
    lpos3 = log_multi.stdout.index("#003")
    assert lpos1 < lpos2 < lpos3
    assert "payload:" in log_multi.stdout

    log_md = runner.invoke(app, ["log", "tutor.opening_turn", "rag.*", "--format", "md"])
    assert log_md.exit_code == 0
    assert "[kind=text]" in log_md.stdout
    assert "[kind=decision]**(HALT TRIGGER)**" not in log_md.stdout
    assert "`[kind=decision]` **(HALT TRIGGER)**" in log_md.stdout

    runs_res = runner.invoke(app, ["runs", "--json"])
    assert runs_res.exit_code == 0
    assert len(json.loads(runs_res.stdout)["runs"]) == 2

    # Clean exit code 1 with Error message when config is invalid across CLI subcommands.
    broken_cfg_project = tmp_path / "bad_cli_cfg"
    broken_cfg_project.mkdir()
    (broken_cfg_project / "pyproject.toml").write_text(
        "[project]\nname = 'bad-cfg'\n[tool.probe.decision]\npass_threshold = 1.0\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(broken_cfg_project)
    for bad_args in (
        ["report"],
        ["log", "tutor.*"],
        ["runs"],
        ["clean"],
    ):
        err_res = runner.invoke(app, bad_args)
        assert err_res.exit_code == 1
        assert "Error:" in err_res.stderr
    monkeypatch.chdir(tmp_path)

    log_no_payload = runner.invoke(app, ["log", "tutor.opening_turn", "--format", "json"])
    assert log_no_payload.exit_code == 0
    assert "payload" not in json.loads(log_no_payload.stdout)["events"][0]
    log_with_payload = runner.invoke(app, ["log", "tutor.opening_turn", "--payload", "--format", "json"])
    assert log_with_payload.exit_code == 0
    assert "payload" in json.loads(log_with_payload.stdout)["events"][0]
    assert runner.invoke(app, ["log", "   "]).exit_code == 1
    assert runner.invoke(app, ["runs", "--limit", "0"]).exit_code != 0
    assert runner.invoke(app, ["clean", "--keep", "-1"]).exit_code != 0

    assert (tmp_path / ".probes" / ".gitignore").read_text(encoding="utf-8") == "runs/\n"
    run_jsonl = runs_dir / f"{run_id}.jsonl"
    assert run_jsonl.is_file()
    parsed_lines = [json.loads(line) for line in run_jsonl.read_text(encoding="utf-8").splitlines()]
    assert parsed_lines[0]["type"] == "run_start"
    assert parsed_lines[-1]["type"] == "run_finish"
    assert isinstance(parsed_lines[1]["payload"], dict)

    # Partial trailing JSON lines and traversal run_ids do not crash readers.
    with run_jsonl.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"event","partial":')
    ro_store = ProbeStore(runs_dir, create=False)
    assert len(ro_store.list_events(run_id)) == 3
    assert ro_store.get_run("../escape") is None
    assert ro_store.list_events("../escape") == []

    clean_res = runner.invoke(app, ["clean", "--keep", "1"])
    assert clean_res.exit_code == 0
    assert "Removed 1 old run(s)" in clean_res.stdout
    assert not (runs_dir / "01RUNOLD000000000000000001.jsonl").exists()
    runs_after = runner.invoke(app, ["runs", "--json"])
    remaining = json.loads(runs_after.stdout)["runs"]
    assert len(remaining) == 1 and remaining[0]["id"] == run_id


def test_auto_derived_ids_zero_imports_gate1_failures_syntax_errors_and_subdir_cli(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "sample-stage4"
            version = "0.1.0"

            [tool.probe.check]
            require_check = true

            [tool.probe.models.decision]
            provider = "http_decision"
            url = "http://mock.local/decision"
            model = "jev"

            [tool.probe.models.text]
            provider = "openai_compat"
            url = "http://mock.local/v1"
            model = "qwen3.5:9b"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    script_py = tmp_path / "script.py"
    script_py.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Greets user politely")
            def top_fn(name: str) -> str:
                @probe("Formats greeting nicely")
                def nested_fn(inner: str) -> str:
                    probe.here("Greeting includes user name", msg=f"Hello {inner}")
                    return f"Hello {inner}"
                return nested_fn(name)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    unrelated_py = tmp_path / "unrelated.py"
    unrelated_py.write_text(
        textwrap.dedent(
            """
            def probe(_msg: str):
                return lambda fn: fn

            @probe("Custom non-probe decorator")
            def custom(x: int) -> int:
                return x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    assert discover_probes([unrelated_py], root=tmp_path) == []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if url.endswith("/chat/completions"):
            sys_msg = body["messages"][0]["content"]
            if "passing_snapshot" in sys_msg:
                user_msg = body["messages"][1]["content"]
                if "Site: here" in user_msg:
                    payload = {
                        "notes": "ok",
                        "passing_snapshot": {"watched": {"msg": "GOOD Hello Ana"}},
                        "failing_snapshot": {"watched": {"msg": "BAD"}},
                    }
                else:
                    payload = {
                        "notes": "ok",
                        "passing_snapshot": {
                            "inputs": {"name": "Ana", "inner": "Ana"},
                            "output": "GOOD Hello Ana",
                        },
                        "failing_snapshot": {
                            "inputs": {"name": "Ana", "inner": "Ana"},
                            "output": "BAD",
                        },
                    }
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Factual check audit."}}]})
        if url.endswith("/decision"):
            if body["probe_id"].endswith("#gate2"):
                return httpx.Response(200, json={"probabilities": [0.95 for _ in body["checks"]]})
            ok = "GOOD" in body["snapshot"] or "Hello Ana" in body["snapshot"]
            return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.10 for _ in body["checks"]]})
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    ok, results = run_probe_check([script_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok is True
    assert [r.probe_id for r in results] == [
        "script.top_fn",
        "script.top_fn.<locals>.nested_fn",
        "script.top_fn.<locals>.nested_fn:L7",
    ]

    # Runtime execution as simulated __main__ must match auto-derived lock IDs without PROBE_SKIP_LOCK.
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.delenv("PROBE_SKIP_LOCK", raising=False)
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    spec = importlib.util.spec_from_file_location("__main__", script_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)
    probe.reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))
    try:
        assert mod.top_fn("Ana") == "Hello Ana"
    finally:
        probe.probe.flush_for_testing()
        probe.reset_for_testing()
        if old_main is not None:
            sys.modules["__main__"] = old_main

    # Invalid Gate 1 patterns: empty probe.here and generator @probe.
    bad_py = tmp_path / "pkg" / "bad_patterns.py"
    bad_py.parent.mkdir()
    bad_py.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Yields integers politely")
            def gen_fn():
                yield 1

            def empty_here() -> None:
                probe.here("Missing watched variables")
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_bad, res_bad = run_probe_check([bad_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok_bad is False
    assert all(r.status == "failed" and r.gate == "gate1" for r in res_bad)
    assert len(res_bad) == 2

    # SyntaxError fails Gate 1 without wiping previously approved lock entries.
    syntax_py = tmp_path / "broken_syntax.py"
    syntax_py.write_text("def broken(:\n", encoding="utf-8")
    ok_syn, res_syn = run_probe_check([syntax_py, script_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok_syn is False
    assert any(r.probe_id == "<parse_error>" and r.gate == "gate1" for r in res_syn)
    lock_after = json.loads((tmp_path / ".probes" / "lock.json").read_text(encoding="utf-8"))["probes"]
    assert "script.top_fn" in lock_after

    # Subdirectory CLI invocation resolves relative paths against CWD and fails on missing paths.
    monkeypatch.chdir(tmp_path / "pkg")
    runner = CliRunner()
    sub_res = runner.invoke(app, ["list", "bad_patterns.py", "--json"])
    assert sub_res.exit_code == 0
    assert len(json.loads(sub_res.stdout)["probes"]) == 2
    missing_res = runner.invoke(app, ["check", "missing_file.py"])
    assert missing_res.exit_code == 1

    # Path traversal in manual fixture names is rejected.
    with pytest.raises(BackendError, match="path separators"):
        _load_manual_fixture(tmp_path / ".probes" / "fixtures", "../secret")


def _stage3_mock_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    body = json.loads(request.content.decode("utf-8"))
    if url.endswith("/chat/completions"):
        sys_msg = body["messages"][0]["content"]
        if "passing_snapshot" in sys_msg:
            user_msg = body["messages"][1]["content"]
            if "Site: here" in user_msg:
                wk_line = next((ln for ln in user_msg.splitlines() if ln.startswith("Watched keys:")), "")
                keys = [k.strip() for k in wk_line.split(":", 1)[1].split(",") if k.strip()] if ":" in wk_line else []
                payload = {
                    "notes": "ok",
                    "passing_snapshot": {"watched": {k: "GOOD ok" for k in keys}},
                    "failing_snapshot": {"watched": {k: "BAD" for k in keys}},
                }
            else:
                # Extract parameter names from Signature line when present.
                sig_line = next((ln for ln in user_msg.splitlines() if ln.startswith("Signature:")), "")
                params_str = sig_line.split("(", 1)[1].rsplit(")", 1)[0] if "(" in sig_line and ")" in sig_line else ""
                inputs = {}
                for chunk in params_str.split(","):
                    pname = chunk.strip().split(":", 1)[0].split("=", 1)[0].strip().lstrip("*")
                    if pname:
                        inputs[pname] = "Ana"
                payload = {
                    "notes": "ok",
                    "passing_snapshot": {"inputs": inputs, "output": "GOOD ok"},
                    "failing_snapshot": {"inputs": inputs, "output": "BAD"},
                }
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Factual check audit."}}]})
    if url.endswith("/decision"):
        if body["probe_id"].endswith("#gate2"):
            return httpx.Response(200, json={"probabilities": [0.95 for _ in body["checks"]]})
        ok = "GOOD" in body["snapshot"]
        return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.10 for _ in body["checks"]]})
    raise AssertionError(f"Unexpected URL: {url}")


def test_stage3_discovery_scoping_namespace_and_stacking(tmp_path: Path, monkeypatch) -> None:
    # 1. PEP 420 namespace package (src/acme/tutor/turn.py with zero __init__.py).
    turn_py = tmp_path / "src" / "acme" / "tutor" / "turn.py"
    turn_py.parent.mkdir(parents=True)
    turn_py.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Responds warmly in Spanish")
            def greet(name: str) -> str:
                return f"Hola {name}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    assert resolve_module_name(turn_py, tmp_path) == "acme.tutor.turn"
    ns_discovered = discover_probes([turn_py], root=tmp_path)
    assert [(d.probe_id, d.file_line, d.param_names) for d in ns_discovered] == [
        ("acme.tutor.turn.greet", "src/acme/tutor/turn.py:4", ("name",))
    ]

    # 2. Stacked decorators under __main__ + generator expression scope isolation.
    stacked_py = tmp_path / "stacked_main.py"
    stacked_py.write_text(
        textwrap.dedent(
            """
            import functools
            from probe import probe

            @probe("Greets cleanly without leaking system tokens")
            @functools.lru_cache(maxsize=4)
            def cached_greet(name: str) -> str:
                gen = (probe.here("Non-empty greeting value", val=f"GOOD {ch}") for ch in [name])
                list(gen)
                return f"GOOD {name}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    disc_stacked = discover_probes([stacked_py], root=tmp_path)
    assert [d.probe_id for d in disc_stacked] == [
        "stacked_main.cached_greet",
        "stacked_main.cached_greet.<locals>.<genexpr>:L7",
    ]
    assert disc_stacked[0].param_names == ("name",)
    assert disc_stacked[0].signature == "def cached_greet(name: str) -> str"

    cfg = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok_st, _ = run_probe_check([stacked_py], config=cfg, transport=httpx.MockTransport(_stage3_mock_handler))
    assert ok_st is True

    monkeypatch.setenv("PROBES", "1")
    monkeypatch.delenv("PROBE_SKIP_LOCK", raising=False)
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    spec = importlib.util.spec_from_file_location("__main__", stacked_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)
    probe.reset_for_testing(config=cfg, transport=httpx.MockTransport(_stage3_mock_handler))
    try:
        assert mod.cached_greet("Ana") == "GOOD Ana"
    finally:
        probe.probe.flush_for_testing()
        probe.reset_for_testing()
        if old_main is not None:
            sys.modules["__main__"] = old_main

    # 3. Lexical shadowing ignores non-probe decorators; @probe on class fails; id=None ok; literal before dup ID.
    scope_py = tmp_path / "scoping_cases.py"
    scope_py.write_text(
        textwrap.dedent(
            """
            from probe import probe

            def outer():
                probe = lambda *a, **kw: (lambda fn: fn)
                @probe("Shadowed local decorator")
                def shadowed_fn(x: str) -> str:
                    return x

            @probe("Class decorators are rejected")
            class BadClass:
                pass

            @probe("Explicit None id uses auto-derived ID", id=None)
            def none_id_fn(topic: str) -> str:
                return topic

            @probe(123, id="dup.case")
            def bad_lit_dup(x: str) -> str:
                return x

            @probe("Valid semantic rule", id="dup.case")
            def good_dup(x: str) -> str:
                return x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_sc, res_sc = run_probe_check([scope_py], config=cfg, transport=httpx.MockTransport(_stage3_mock_handler))
    assert ok_sc is False
    by_id = {r.probe_id: r for r in res_sc}
    assert "scoping_cases.outer.<locals>.shadowed_fn" not in by_id
    assert by_id["scoping_cases.BadClass"].gate == "gate1"
    assert "classes" in by_id["scoping_cases.BadClass"].reason
    assert by_id["scoping_cases.none_id_fn"].status == "passed"
    dup_res = [r for r in res_sc if r.probe_id == "dup.case"]
    assert len(dup_res) == 2
    assert any("literal" in r.reason or "string literal" in r.reason for r in dup_res)
    assert any("Duplicate probe_id" in r.reason for r in dup_res)


def test_stage3_pruning_moves_fixtures_list_and_verify_lock(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "stage3-app"',
                "",
                "[tool.probe.models.decision]",
                'provider = "http_decision"',
                'url = "http://mock.local/decision"',
                'model = "jev"',
                "",
                "[tool.probe.models.text]",
                'provider = "openai_compat"',
                'url = "http://mock.local/v1"',
                'model = "qwen3.5:9b"',
                "",
                "[tool.probe.check]",
                "require_check = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pkg = tmp_path / "pkg"
    other = tmp_path / "other"
    pkg.mkdir()
    other.mkdir()
    a_py = pkg / "a.py"
    old_py = pkg / "old.py"
    keep_py = other / "keep.py"
    a_py.write_text(
        'from probe import probe\n'
        '@probe("Responds politely", id="p.a")\n'
        'def fa(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )
    old_py.write_text(
        'from probe import probe\n'
        '@probe("Responds politely", id="p.old")\n'
        'def fold(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )
    keep_py.write_text(
        'from probe import probe\n'
        '@probe("Responds politely", id="p.keep")\n'
        'def fkeep(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )

    cfg = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    lock_path = cfg.resolve_lock_path()
    transport = httpx.MockTransport(_stage3_mock_handler)
    ok_all, _ = run_probe_check(config=cfg, transport=transport)
    assert ok_all is True

    # 1. Directory scan prunes deleted pkg/old.py while keeping other/keep.py.
    old_py.unlink()
    ok_dir, _ = run_probe_check([pkg], config=cfg, transport=transport)
    assert ok_dir is True
    locks = json.loads(lock_path.read_text(encoding="utf-8"))["probes"]
    assert "p.old" not in locks and "p.a" in locks and "p.keep" in locks
    assert locks["p.a"]["param_names"] == ["x"]

    # 2. Explicit CLI check on deleted locked file prunes it; non-existent untracked file still exits 1.
    temp_py = pkg / "temp_del.py"
    temp_py.write_text(
        'from probe import probe\n@probe("Responds politely", id="p.temp")\ndef ftmp(x: str) -> str:\n    return x\n',
        encoding="utf-8",
    )
    assert run_probe_check([temp_py], config=cfg, transport=transport)[0] is True
    assert "p.temp" in json.loads(lock_path.read_text(encoding="utf-8"))["probes"]
    temp_py.unlink()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    del_res = runner.invoke(app, ["check", "pkg/temp_del.py"])
    assert del_res.exit_code == 0, del_res.stdout + del_res.stderr
    assert "p.temp" not in json.loads(lock_path.read_text(encoding="utf-8"))["probes"]
    assert runner.invoke(app, ["check", "never_existed.py"]).exit_code == 1

    # 3. Parse-error file protects its locked entry against both pruning and cross-file theft.
    a_py.write_text("def broken(:\n", encoding="utf-8")
    ok_err, _ = run_probe_check(config=cfg, transport=transport)
    assert ok_err is False
    assert "p.a" in json.loads(lock_path.read_text(encoding="utf-8"))["probes"]
    a_py.write_text(
        'from probe import probe\n'
        '@probe("Responds politely", id="p.a")\n'
        'def fa(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )

    # 4. Partial scan allows moving p.keep from existing other/keep.py -> pkg/a.py, but still blocks true duplicates.
    keep_py.write_text("# moved out\n", encoding="utf-8")
    a_py.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Responds politely", id="p.a")
            def fa(x: str) -> str:
                return "GOOD " + x

            @probe("Responds politely", id="p.keep")
            def moved_keep(x: str) -> str:
                return "GOOD " + x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_move, _ = run_probe_check([a_py], config=cfg, transport=transport)
    assert ok_move is True
    locks_after_move = json.loads(lock_path.read_text(encoding="utf-8"))["probes"]
    assert locks_after_move["p.keep"]["file_line"].startswith("pkg/a.py:")

    # Duplicate probe_id in existing keep_py blocks checking keep_py against a_py.
    keep_py.write_text(
        'from probe import probe\n'
        '@probe("Responds politely", id="p.keep")\n'
        'def dup_keep(x: str) -> str:\n    return x\n',
        encoding="utf-8",
    )
    ok_dup_partial, res_dup_partial = run_probe_check([keep_py], config=cfg, transport=transport)
    assert ok_dup_partial is False
    assert res_dup_partial[0].gate == "gate1"
    assert "already locked" in res_dup_partial[0].reason
    keep_py.write_text("# clean again\n", encoding="utf-8")

    # 5. Manual fixture hash disambiguation, wrong probe_id rejection, and strict Gate 3 snapshot validation.
    fixtures_dir = tmp_path / ".probes" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    hashed_name = f"pkg_colon-{_sha256_text('pkg:colon')[:6]}.json"
    (fixtures_dir / hashed_name).write_text(
        json.dumps(
            {
                "probe_id": "pkg:colon",
                "pass": {"inputs": {"x": "hi"}, "output": "GOOD hi"},
                "fail": {"inputs": {"x": "hi"}, "output": "BAD"},
            }
        ),
        encoding="utf-8",
    )
    assert _load_manual_fixture(fixtures_dir, "pkg:colon") is not None
    (fixtures_dir / "wrong_pid.json").write_text(
        json.dumps(
            {
                "probe_id": "other.id",
                "pass": {"inputs": {"x": "hi"}, "output": "GOOD"},
                "fail": {"inputs": {"x": "hi"}, "output": "BAD"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BackendError, match="does not match"):
        _load_manual_fixture(fixtures_dir, "wrong_pid")

    dec_item = discover_probes([a_py], root=tmp_path)[0]
    with pytest.raises(BackendError, match="must not use 'exception'"):
        _validate_gate3_snapshot('{"inputs": {"x": "a"}, "exception": "boom"}', item=dec_item, label="pass")
    with pytest.raises(BackendError, match="missing required input"):
        _validate_gate3_snapshot('{"inputs": {}, "output": "GOOD"}', item=dec_item, label="pass")

    here_py = tmp_path / "here_item.py"
    here_py.write_text(
        'from probe import probe\ndef f(a, b):\n    probe.here("Warm reply", id="h.val", a=a, b=b)\n',
        encoding="utf-8",
    )
    here_item = discover_probes([here_py], root=tmp_path)[0]
    with pytest.raises(BackendError, match="must not include 'output'"):
        _validate_gate3_snapshot('{"watched": {"a": 1, "b": 2}, "output": "no"}', item=here_item, label="pass")
    with pytest.raises(BackendError, match="unexpected watched key"):
        _validate_gate3_snapshot('{"watched": {"a": 1, "b": 2, "c": 3}}', item=here_item, label="pass")

    # 6. probe list: invalid status, threshold change invalidation, and SyntaxError exit 1.
    list_ok = runner.invoke(app, ["list", "pkg/a.py", "--json"])
    assert list_ok.exit_code == 0
    rows = json.loads(list_ok.stdout)["probes"]
    assert all(r["lock_status"] == "locked" for r in rows)

    (tmp_path / "pyproject.toml").write_text(
        (tmp_path / "pyproject.toml").read_text(encoding="utf-8") + "\n[tool.probe.decision]\npass_threshold = 0.65\n",
        encoding="utf-8",
    )
    list_thresh = runner.invoke(app, ["list", "pkg/a.py", "--json"])
    assert list_thresh.exit_code == 0
    assert all(r["lock_status"] != "locked" for r in json.loads(list_thresh.stdout)["probes"])

    dup_file = tmp_path / "dups.py"
    dup_file.write_text(
        'from probe import probe\n@probe(123, id="d1")\ndef f1(): pass\n@probe("ok", id="d1")\ndef f2(): pass\n',
        encoding="utf-8",
    )
    list_inv = runner.invoke(app, ["list", "dups.py", "--json"])
    assert list_inv.exit_code == 0
    assert [r["lock_status"] for r in json.loads(list_inv.stdout)["probes"]] == ["invalid", "invalid"]

    syn_file = tmp_path / "syn_bad.py"
    syn_file.write_text("def x(:\n", encoding="utf-8")
    assert runner.invoke(app, ["list", "syn_bad.py"]).exit_code == 1

    # 7. Runtime verify_lock rejects param_names mismatch and wrong-file spoofing.
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.delenv("PROBE_SKIP_LOCK", raising=False)
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    probe.reset_for_testing(config=cfg, transport=transport)
    try:
        rt = probe._get_runtime()
        with pytest.raises(ProbeHaltError):
            rt.verify_lock(
                probe_id="p.a",
                probe_type=ProbeType.decision,
                checks=("Responds politely",),
                call_seq=1,
                elapsed_ms=1.0,
                started_at="2026-11-18T10:00:00Z",
                site="decorator",
                file_line="pkg/a.py:3",
                payload_json="{}",
                watched_keys=(),
                param_names=("renamed_param",),
            )
    finally:
        probe.probe.flush_for_testing()
        probe.reset_for_testing()

    probe.reset_for_testing(config=cfg, transport=transport)
    try:
        rt2 = probe._get_runtime()
        with pytest.raises(ProbeHaltError):
            rt2.verify_lock(
                probe_id="p.a",
                probe_type=ProbeType.decision,
                checks=("Responds politely",),
                call_seq=1,
                elapsed_ms=1.0,
                started_at="2026-11-18T10:00:00Z",
                site="decorator",
                file_line="other/spoof.py:99",
                payload_json="{}",
                watched_keys=(),
                param_names=("x",),
            )
    finally:
        probe.probe.flush_for_testing()
        probe.reset_for_testing()


def test_stage4_gate3_synthetic_retry_and_manual_fixture_no_retry(tmp_path: Path) -> None:
    target_py = tmp_path / "retry_demo.py"
    target_py.write_text(
        textwrap.dedent(
            """
            from probe import probe

            @probe("Responds warmly in Spanish without leaking answers", id="retry.syn")
            def syn_fn(topic: str) -> str:
                return topic

            @probe("Stays grounded in context", id="retry.manual")
            def manual_fn(topic: str) -> str:
                return topic
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    fixtures_dir = tmp_path / ".probes" / "fixtures"
    fixtures_dir.mkdir(parents=True)
    # Manual fixture that fails calibration (both pass & fail look BAD) — must NOT retry!
    (fixtures_dir / "retry.manual.json").write_text(
        json.dumps(
            {
                "pass": {"inputs": {"topic": "coffee"}, "output": "WEAK_MANUAL"},
                "fail": {"inputs": {"topic": "coffee"}, "output": "WEAK_FAIL"},
            }
        ),
        encoding="utf-8",
    )

    gate3_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if url.endswith("/chat/completions"):
            sys_msg = body["messages"][0]["content"]
            if "passing_snapshot" in sys_msg:
                gate3_calls.append(body)
                if len(gate3_calls) == 1:
                    # Attempt 1: weak passing_snapshot causes calibration failure.
                    payload = {
                        "notes": "attempt 1",
                        "passing_snapshot": {"inputs": {"topic": "coffee"}, "output": "WEAK_SYN"},
                        "failing_snapshot": {"inputs": {"topic": "coffee"}, "output": "BAD_SYN"},
                    }
                else:
                    # Attempt 2: fixes calibration after receiving retry_feedback & temperature=0.3.
                    payload = {
                        "notes": "attempt 2",
                        "passing_snapshot": {"inputs": {"topic": "coffee"}, "output": "GOOD Hola!"},
                        "failing_snapshot": {"inputs": {"topic": "coffee"}, "output": "BAD_SYN"},
                    }
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Factual check audit."}}]})
        if url.endswith("/decision"):
            if body["probe_id"].endswith("#gate2"):
                return httpx.Response(200, json={"probabilities": [0.95 for _ in body["checks"]]})
            ok = "GOOD" in body["snapshot"]
            return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.15 for _ in body["checks"]]})
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        domain_context="Custom pedagogical tag rules.",
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    ok, results = run_probe_check([target_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok is False  # retry.manual failed as intended
    by_id = {r.probe_id: r for r in results}
    assert by_id["retry.syn"].status == "passed"
    assert by_id["retry.manual"].status == "failed" and by_id["retry.manual"].gate == "gate3"
    # Exactly 2 Gate 3 synthesis calls: attempt 1 + 1x retry for retry.syn, 0 synthesis calls for retry.manual.
    assert len(gate3_calls) == 2
    assert gate3_calls[0]["temperature"] == 0.0
    assert "<previous_attempt_error>" not in gate3_calls[0]["messages"][1]["content"]
    assert "<domain_context>" in gate3_calls[0]["messages"][1]["content"]
    assert gate3_calls[1]["temperature"] == 0.3
    assert "<previous_attempt_error>" in gate3_calls[1]["messages"][1]["content"]
    assert "failed checks" in gate3_calls[1]["messages"][1]["content"]
