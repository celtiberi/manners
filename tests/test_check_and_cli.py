"""Tests for AST discovery, manners check Gates 1-3, lockfile caching, and all CLI commands."""

from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import textwrap
import threading
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import manners
from manners import MannersHaltError
from manners.backends import BackendError, CheckScore
from manners.check import (
    _load_manual_fixture,
    _sha256_text,
    _validate_gate3_snapshot,
    discover_manners,
    run_manners_check,
)
from manners.cli import app
from manners.config import (
    CheckConfig,
    DecisionConfig,
    MannersConfig,
    ModelEndpointConfig,
    RuntimeConfig,
    TextConfig,
)
from manners.runtime import MannersType, resolve_module_name
from manners.store import MannersStore


def _write_sample_project(root: Path) -> Path:
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "sample-app"
            version = "0.1.0"

            [tool.manners.models.decision]
            provider = "http_decision"
            url = "http://mock.local/decision"
            model = "local-decision"

            [tool.manners.models.text]
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
            from manners import manners

            @manners(
                [
                    "Tutor opens with a short Spanish greeting",
                    "Tutor does NOT give away the target Spanish sentence",
                ],
                id="tutor.opening_turn",
            )
            def generate_opening_turn(lesson: str) -> str:
                manners.here(
                    "Draft stays grounded in lesson vocabulary",
                    id="tutor.draft",
                    kind=manners.type.text,
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
    discovered = discover_manners(root=tmp_path)
    assert [(d.check_id, d.site, d.check_type.value) for d in discovered] == [
        ("tutor.opening_turn", "decorator", "decision"),
        ("tutor.draft", "here", "text"),
    ]

    # Add a manual fixture for tutor.draft so Gate 3 tests both fixture and synthetic paths.
    fixtures_dir = tmp_path / ".manners" / "fixtures"
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
            if "audit @manners and manners.here" in sys_msg:
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
            if body["check_id"].endswith("#gate2"):
                return httpx.Response(200, json={"probabilities": [0.05 for _ in body["checks"]]})
            snap = body["snapshot"]
            ok = "GOOD" in snap
            probs = [0.95 if ok else 0.10 for _ in body["checks"]]
            return httpx.Response(200, json={"probabilities": probs})
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        text=TextConfig(),
        check=CheckConfig(),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    ok, results = run_manners_check(config=cfg, transport=httpx.MockTransport(handler))
    assert ok is True
    assert [r.status for r in results] == ["passed", "passed"]
    first_calls = http_calls["n"]
    assert first_calls > 0
    assert (tmp_path / ".manners" / "lock.json").is_file()

    # Second run skips Gates 2-3 cleanly via cache (zero HTTP calls).
    ok2, results2 = run_manners_check(config=cfg, transport=httpx.MockTransport(handler))
    assert ok2 is True
    assert [r.status for r in results2] == ["cached", "cached"]
    assert http_calls["n"] == first_calls

    # --force re-runs Gates 2-3.
    ok3, results3 = run_manners_check(force=True, config=cfg, transport=httpx.MockTransport(handler))
    assert ok3 is True
    assert [r.status for r in results3] == ["passed", "passed"]
    assert http_calls["n"] > first_calls


def test_gate1_and_gate2_rejections(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners(["x > 0", "y > 0", "z > 0"], id="dup.id")
            def fn_a(x: int, y: int, z: int) -> int:
                return x

            @manners("still dup", id="dup.id")
            def fn_b() -> int:
                return 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen"),
        check=CheckConfig(max_checks=2, max_check_chars=200),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok, results = run_manners_check(config=cfg)
    assert ok is False
    assert all(r.status == "failed" and r.gate == "gate1" for r in results)

    # Gate 2 rejection on deterministic math check.
    (tmp_path / "bad.py").write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("Sum equals a + b", id="math.add")
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
            return httpx.Response(200, json={"probabilities": [0.95, 0.05, 0.05]})
        raise AssertionError(f"Unexpected URL: {url}")

    ok_g2, res_g2 = run_manners_check(config=cfg, transport=httpx.MockTransport(reject_handler))
    assert ok_g2 is False
    assert len(res_g2) == 1
    assert res_g2[0].gate == "gate2"
    assert "assert" in res_g2[0].reason.lower()


def test_cli_list_report_log_runs_clean(tmp_path: Path, monkeypatch) -> None:
    _write_sample_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    runs_dir = tmp_path / ".manners" / "runs"
    store = MannersStore(runs_dir, create=True)
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
            check_id="tutor.opening_turn",
            site="decorator",
            check_type="decision",
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
            check_id="rag.draft_answer",
            site="here",
            check_type="text",
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
            check_id="tutor.opening_turn",
            site="decorator",
            check_type="decision",
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
    assert [p["check_id"] for p in list_data["manners"]] == ["tutor.opening_turn", "tutor.draft"]
    assert all(p["lock_status"] == "unlocked" for p in list_data["manners"])

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
        "[project]\nname = 'bad-cfg'\n[tool.manners.decision]\npass_threshold = 1.0\n",
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

    assert (tmp_path / ".manners" / ".gitignore").read_text(encoding="utf-8") == "runs/\n"
    run_jsonl = runs_dir / f"{run_id}.jsonl"
    assert run_jsonl.is_file()
    parsed_lines = [json.loads(line) for line in run_jsonl.read_text(encoding="utf-8").splitlines()]
    assert parsed_lines[0]["type"] == "run_start"
    assert parsed_lines[-1]["type"] == "run_finish"
    assert isinstance(parsed_lines[1]["payload"], dict)

    # Partial trailing JSON lines and traversal run_ids do not crash readers.
    with run_jsonl.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"event","partial":')
    ro_store = MannersStore(runs_dir, create=False)
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

            [tool.manners.check]
            require_check = true

            [tool.manners.models.decision]
            provider = "http_decision"
            url = "http://mock.local/decision"
            model = "jev"

            [tool.manners.models.text]
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
            from manners import manners

            @manners("Greets user politely")
            def top_fn(name: str) -> str:
                @manners("Formats greeting nicely")
                def nested_fn(inner: str) -> str:
                    manners.here("Greeting includes user name", msg=f"Hello {inner}")
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
            def manners(_msg: str):
                return lambda fn: fn

            @manners("Custom non-manners decorator")
            def custom(x: int) -> int:
                return x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    assert discover_manners([unrelated_py], root=tmp_path) == []

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
            if body["check_id"].endswith("#gate2"):
                return httpx.Response(200, json={"probabilities": [0.05 for _ in body["checks"]]})
            ok = "GOOD" in body["snapshot"] or "Hello Ana" in body["snapshot"]
            return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.10 for _ in body["checks"]]})
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    ok, results = run_manners_check([script_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok is True
    assert [r.check_id for r in results] == [
        "script.top_fn",
        "script.top_fn.<locals>.nested_fn",
        "script.top_fn.<locals>.nested_fn:L7",
    ]

    # Runtime execution as simulated __main__ must match auto-derived lock IDs without MANNERS_SKIP_LOCK.
    monkeypatch.setenv("MANNERS", "1")
    monkeypatch.delenv("MANNERS_SKIP_LOCK", raising=False)
    monkeypatch.setenv("MANNERS_NO_PDB", "1")
    spec = importlib.util.spec_from_file_location("__main__", script_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)
    manners.reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))
    try:
        assert mod.top_fn("Ana") == "Hello Ana"
    finally:
        manners.manners.flush_for_testing()
        manners.reset_for_testing()
        if old_main is not None:
            sys.modules["__main__"] = old_main

    # Invalid Gate 1 patterns: empty manners.here and generator @manners.
    bad_py = tmp_path / "pkg" / "bad_patterns.py"
    bad_py.parent.mkdir()
    bad_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("Yields integers politely")
            def gen_fn():
                yield 1

            def empty_here() -> None:
                manners.here("Missing watched variables")
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_bad, res_bad = run_manners_check([bad_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok_bad is False
    assert all(r.status == "failed" and r.gate == "gate1" for r in res_bad)
    assert len(res_bad) == 2

    # SyntaxError fails Gate 1 without wiping previously approved lock entries.
    syntax_py = tmp_path / "broken_syntax.py"
    syntax_py.write_text("def broken(:\n", encoding="utf-8")
    ok_syn, res_syn = run_manners_check([syntax_py, script_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok_syn is False
    assert any(r.check_id == "<parse_error>" and r.gate == "gate1" for r in res_syn)
    lock_after = json.loads((tmp_path / ".manners" / "lock.json").read_text(encoding="utf-8"))["manners"]
    assert "script.top_fn" in lock_after

    # Subdirectory CLI invocation resolves relative paths against CWD and fails on missing paths.
    monkeypatch.chdir(tmp_path / "pkg")
    runner = CliRunner()
    sub_res = runner.invoke(app, ["list", "bad_patterns.py", "--json"])
    assert sub_res.exit_code == 0
    assert len(json.loads(sub_res.stdout)["manners"]) == 2
    missing_res = runner.invoke(app, ["check", "missing_file.py"])
    assert missing_res.exit_code == 1

    # Path traversal in manual fixture names is rejected.
    with pytest.raises(BackendError, match="path separators"):
        _load_manual_fixture(tmp_path / ".manners" / "fixtures", "../secret")


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
        if body["check_id"].endswith("#gate2"):
            return httpx.Response(200, json={"probabilities": [0.05 for _ in body["checks"]]})
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
            from manners import manners

            @manners("Responds warmly in Spanish")
            def greet(name: str) -> str:
                return f"Hola {name}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    assert resolve_module_name(turn_py, tmp_path) == "acme.tutor.turn"
    ns_discovered = discover_manners([turn_py], root=tmp_path)
    assert [(d.check_id, d.file_line, d.param_names) for d in ns_discovered] == [
        ("acme.tutor.turn.greet", "src/acme/tutor/turn.py:4", ("name",))
    ]

    # 2. Stacked decorators under __main__ + generator expression scope isolation.
    stacked_py = tmp_path / "stacked_main.py"
    stacked_py.write_text(
        textwrap.dedent(
            """
            import functools
            from manners import manners

            @manners("Greets cleanly without leaking system tokens")
            @functools.lru_cache(maxsize=4)
            def cached_greet(name: str) -> str:
                gen = (manners.here("Non-empty greeting value", val=f"GOOD {ch}") for ch in [name])
                list(gen)
                return f"GOOD {name}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    disc_stacked = discover_manners([stacked_py], root=tmp_path)
    assert [d.check_id for d in disc_stacked] == [
        "stacked_main.cached_greet",
        "stacked_main.cached_greet.<locals>.<genexpr>:L7",
    ]
    assert disc_stacked[0].param_names == ("name",)
    assert disc_stacked[0].signature == "def cached_greet(name: str) -> str"

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok_st, _ = run_manners_check([stacked_py], config=cfg, transport=httpx.MockTransport(_stage3_mock_handler))
    assert ok_st is True

    monkeypatch.setenv("MANNERS", "1")
    monkeypatch.delenv("MANNERS_SKIP_LOCK", raising=False)
    monkeypatch.setenv("MANNERS_NO_PDB", "1")
    spec = importlib.util.spec_from_file_location("__main__", stacked_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    old_main = sys.modules.get("__main__")
    sys.modules["__main__"] = mod
    spec.loader.exec_module(mod)
    manners.reset_for_testing(config=cfg, transport=httpx.MockTransport(_stage3_mock_handler))
    try:
        assert mod.cached_greet("Ana") == "GOOD Ana"
    finally:
        manners.manners.flush_for_testing()
        manners.reset_for_testing()
        if old_main is not None:
            sys.modules["__main__"] = old_main

    # 3. Lexical shadowing ignores non-manners decorators; @manners on class fails; id=None ok; literal before dup ID.
    scope_py = tmp_path / "scoping_cases.py"
    scope_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            def outer():
                manners = lambda *a, **kw: (lambda fn: fn)
                @manners("Shadowed local decorator")
                def shadowed_fn(x: str) -> str:
                    return x

            @manners("Class decorators are rejected")
            class BadClass:
                pass

            @manners("Explicit None id uses auto-derived ID", id=None)
            def none_id_fn(topic: str) -> str:
                return topic

            @manners(123, id="dup.case")
            def bad_lit_dup(x: str) -> str:
                return x

            @manners("Valid semantic rule", id="dup.case")
            def good_dup(x: str) -> str:
                return x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_sc, res_sc = run_manners_check([scope_py], config=cfg, transport=httpx.MockTransport(_stage3_mock_handler))
    assert ok_sc is False
    by_id = {r.check_id: r for r in res_sc}
    assert "scoping_cases.outer.<locals>.shadowed_fn" not in by_id
    assert by_id["scoping_cases.BadClass"].gate == "gate1"
    assert "classes" in by_id["scoping_cases.BadClass"].reason
    assert by_id["scoping_cases.none_id_fn"].status == "passed"
    dup_res = [r for r in res_sc if r.check_id == "dup.case"]
    assert len(dup_res) == 2
    assert any("literal" in r.reason or "string literal" in r.reason for r in dup_res)
    assert any("Duplicate check_id" in r.reason for r in dup_res)


def test_stage3_pruning_moves_fixtures_list_and_verify_lock(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "stage3-app"',
                "",
                "[tool.manners.models.decision]",
                'provider = "http_decision"',
                'url = "http://mock.local/decision"',
                'model = "jev"',
                "",
                "[tool.manners.models.text]",
                'provider = "openai_compat"',
                'url = "http://mock.local/v1"',
                'model = "qwen3.5:9b"',
                "",
                "[tool.manners.check]",
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
        'from manners import manners\n'
        '@manners("Responds politely", id="p.a")\n'
        'def fa(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )
    old_py.write_text(
        'from manners import manners\n'
        '@manners("Responds politely", id="p.old")\n'
        'def fold(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )
    keep_py.write_text(
        'from manners import manners\n'
        '@manners("Responds politely", id="p.keep")\n'
        'def fkeep(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    lock_path = cfg.resolve_lock_path()
    transport = httpx.MockTransport(_stage3_mock_handler)
    ok_all, _ = run_manners_check(config=cfg, transport=transport)
    assert ok_all is True

    # 1. Directory scan prunes deleted pkg/old.py while keeping other/keep.py.
    old_py.unlink()
    ok_dir, _ = run_manners_check([pkg], config=cfg, transport=transport)
    assert ok_dir is True
    locks = json.loads(lock_path.read_text(encoding="utf-8"))["manners"]
    assert "p.old" not in locks and "p.a" in locks and "p.keep" in locks
    assert locks["p.a"]["param_names"] == ["x"]

    # 2. Explicit CLI check on deleted locked file prunes it; non-existent untracked file still exits 1.
    temp_py = pkg / "temp_del.py"
    temp_py.write_text(
        'from manners import manners\n@manners("Responds politely", id="p.temp")\n'
        "def ftmp(x: str) -> str:\n    return x\n",
        encoding="utf-8",
    )
    assert run_manners_check([temp_py], config=cfg, transport=transport)[0] is True
    assert "p.temp" in json.loads(lock_path.read_text(encoding="utf-8"))["manners"]
    temp_py.unlink()
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    del_res = runner.invoke(app, ["check", "pkg/temp_del.py"])
    assert del_res.exit_code == 0, del_res.stdout + del_res.stderr
    assert "p.temp" not in json.loads(lock_path.read_text(encoding="utf-8"))["manners"]
    assert runner.invoke(app, ["check", "never_existed.py"]).exit_code == 1

    # 3. Parse-error file protects its locked entry against both pruning and cross-file theft.
    a_py.write_text("def broken(:\n", encoding="utf-8")
    ok_err, _ = run_manners_check(config=cfg, transport=transport)
    assert ok_err is False
    assert "p.a" in json.loads(lock_path.read_text(encoding="utf-8"))["manners"]
    a_py.write_text(
        'from manners import manners\n'
        '@manners("Responds politely", id="p.a")\n'
        'def fa(x: str) -> str:\n    return "GOOD " + x\n',
        encoding="utf-8",
    )

    # 4. Partial scan allows moving p.keep from existing other/keep.py -> pkg/a.py, but still blocks true duplicates.
    keep_py.write_text("# moved out\n", encoding="utf-8")
    a_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("Responds politely", id="p.a")
            def fa(x: str) -> str:
                return "GOOD " + x

            @manners("Responds politely", id="p.keep")
            def moved_keep(x: str) -> str:
                return "GOOD " + x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_move, _ = run_manners_check([a_py], config=cfg, transport=transport)
    assert ok_move is True
    locks_after_move = json.loads(lock_path.read_text(encoding="utf-8"))["manners"]
    assert locks_after_move["p.keep"]["file_line"].startswith("pkg/a.py:")

    # Duplicate check_id in existing keep_py blocks checking keep_py against a_py.
    keep_py.write_text(
        'from manners import manners\n'
        '@manners("Responds politely", id="p.keep")\n'
        'def dup_keep(x: str) -> str:\n    return x\n',
        encoding="utf-8",
    )
    ok_dup_partial, res_dup_partial = run_manners_check([keep_py], config=cfg, transport=transport)
    assert ok_dup_partial is False
    assert res_dup_partial[0].gate == "gate1"
    assert "already locked" in res_dup_partial[0].reason
    keep_py.write_text("# clean again\n", encoding="utf-8")

    # 5. Manual fixture hash disambiguation, wrong check_id rejection, and strict Gate 3 snapshot validation.
    fixtures_dir = tmp_path / ".manners" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    hashed_name = f"pkg_colon-{_sha256_text('pkg:colon')[:6]}.json"
    (fixtures_dir / hashed_name).write_text(
        json.dumps(
            {
                "check_id": "pkg:colon",
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
                "check_id": "other.id",
                "pass": {"inputs": {"x": "hi"}, "output": "GOOD"},
                "fail": {"inputs": {"x": "hi"}, "output": "BAD"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BackendError, match="does not match"):
        _load_manual_fixture(fixtures_dir, "wrong_pid")

    dec_item = discover_manners([a_py], root=tmp_path)[0]
    with pytest.raises(BackendError, match="must not use 'exception'"):
        _validate_gate3_snapshot('{"inputs": {"x": "a"}, "exception": "boom"}', item=dec_item, label="pass")
    with pytest.raises(BackendError, match="missing required input"):
        _validate_gate3_snapshot('{"inputs": {}, "output": "GOOD"}', item=dec_item, label="pass")

    here_py = tmp_path / "here_item.py"
    here_py.write_text(
        'from manners import manners\ndef f(a, b):\n    manners.here("Warm reply", id="h.val", a=a, b=b)\n',
        encoding="utf-8",
    )
    here_item = discover_manners([here_py], root=tmp_path)[0]
    with pytest.raises(BackendError, match="must not include 'output'"):
        _validate_gate3_snapshot('{"watched": {"a": 1, "b": 2}, "output": "no"}', item=here_item, label="pass")
    with pytest.raises(BackendError, match="unexpected watched key"):
        _validate_gate3_snapshot('{"watched": {"a": 1, "b": 2, "c": 3}}', item=here_item, label="pass")

    # 6. manners list: invalid status, threshold change invalidation, and SyntaxError exit 1.
    list_ok = runner.invoke(app, ["list", "pkg/a.py", "--json"])
    assert list_ok.exit_code == 0
    rows = json.loads(list_ok.stdout)["manners"]
    assert all(r["lock_status"] == "locked" for r in rows)

    (tmp_path / "pyproject.toml").write_text(
        (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
        + "\n[tool.manners.decision]\npass_threshold = 0.65\n",
        encoding="utf-8",
    )
    list_thresh = runner.invoke(app, ["list", "pkg/a.py", "--json"])
    assert list_thresh.exit_code == 0
    assert all(r["lock_status"] != "locked" for r in json.loads(list_thresh.stdout)["manners"])

    dup_file = tmp_path / "dups.py"
    dup_file.write_text(
        'from manners import manners\n@manners(123, id="d1")\ndef f1(): pass\n'
        '@manners("ok", id="d1")\ndef f2(): pass\n',
        encoding="utf-8",
    )
    list_inv = runner.invoke(app, ["list", "dups.py", "--json"])
    assert list_inv.exit_code == 0
    assert [r["lock_status"] for r in json.loads(list_inv.stdout)["manners"]] == ["invalid", "invalid"]

    syn_file = tmp_path / "syn_bad.py"
    syn_file.write_text("def x(:\n", encoding="utf-8")
    assert runner.invoke(app, ["list", "syn_bad.py"]).exit_code == 1

    # 7. Runtime verify_lock rejects param_names mismatch and wrong-file spoofing.
    monkeypatch.setenv("MANNERS", "1")
    monkeypatch.delenv("MANNERS_SKIP_LOCK", raising=False)
    monkeypatch.setenv("MANNERS_NO_PDB", "1")
    manners.reset_for_testing(config=cfg, transport=transport)
    try:
        rt = manners._get_runtime()
        with pytest.raises(MannersHaltError):
            rt.verify_lock(
                check_id="p.a",
                check_type=MannersType.decision,
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
        manners.manners.flush_for_testing()
        manners.reset_for_testing()

    manners.reset_for_testing(config=cfg, transport=transport)
    try:
        rt2 = manners._get_runtime()
        with pytest.raises(MannersHaltError):
            rt2.verify_lock(
                check_id="p.a",
                check_type=MannersType.decision,
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
        manners.manners.flush_for_testing()
        manners.reset_for_testing()


def test_stage4_gate3_synthetic_retry_and_manual_fixture_no_retry(tmp_path: Path) -> None:
    target_py = tmp_path / "retry_demo.py"
    target_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("Responds warmly in Spanish without leaking answers", id="retry.syn")
            def syn_fn(topic: str) -> str:
                return topic

            @manners("Stays grounded in context", id="retry.manual")
            def manual_fn(topic: str) -> str:
                return topic
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    fixtures_dir = tmp_path / ".manners" / "fixtures"
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
            if body["check_id"].endswith("#gate2"):
                return httpx.Response(200, json={"probabilities": [0.05 for _ in body["checks"]]})
            ok = "GOOD" in body["snapshot"]
            return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.15 for _ in body["checks"]]})
        raise AssertionError(f"Unexpected URL: {url}")

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        domain_context="Custom pedagogical tag rules.",
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    ok, results = run_manners_check([target_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok is False  # retry.manual failed as intended
    by_id = {r.check_id: r for r in results}
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


def test_v03_parallel_manners_check_and_no_body_lock_churn(tmp_path: Path, monkeypatch) -> None:
    target_py = tmp_path / "app_mod.py"
    target_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("First check returns warm greeting", id="par.p1")
            def p1(name: str) -> str:
                \"\"\"Greet the learner.\"\"\"
                return f"GOOD {name}"

            @manners("Second check returns warm farewell", id="par.p2")
            def p2(name: str) -> str:
                \"\"\"Farewell message.\"\"\"
                return f"GOOD {name}"

            def p3_wrap(draft: str) -> str:
                manners.here("Third check watches draft inline", id="par.p3", draft=draft)
                return draft
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    # Barrier requires 3 uncached Gate 2 calls to arrive concurrently -> would deadlock if sequential!
    gate2_barrier = threading.Barrier(3, timeout=3.0)
    http_lock = threading.Lock()
    http_calls = {"n": 0}

    def parallel_handler(request: httpx.Request) -> httpx.Response:
        with http_lock:
            http_calls["n"] += 1
        resp = _stage3_mock_handler(request)
        if str(request.url).endswith("/chat/completions"):
            body = json.loads(request.content.decode("utf-8"))
            if "passing_snapshot" not in body["messages"][0]["content"]:
                gate2_barrier.wait()
        return resp

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True, max_workers=4),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok, results = run_manners_check([target_py], config=cfg, transport=httpx.MockTransport(parallel_handler))
    assert ok is True
    assert [(r.check_id, r.status) for r in results] == [
        ("par.p1", "passed"),
        ("par.p2", "passed"),
        ("par.p3", "passed"),
    ]

    # Edit internal body / logging and surrounding comments without changing signature, docstring, or watched_keys.
    target_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("First check returns warm greeting", id="par.p1")
            def p1(name: str) -> str:
                \"\"\"Greet the learner.\"\"\"
                tmp = name.strip()
                # unrelated comment and refactor inside body
                return f"GOOD {tmp}!"

            @manners("Second check returns warm farewell", id="par.p2")
            def p2(name: str) -> str:
                \"\"\"Farewell message.\"\"\"
                return f"GOOD bye {name}"

            def p3_wrap(draft: str) -> str:
                # extra comment line above manners.here
                note = draft.strip()
                manners.here("Third check watches draft inline", id="par.p3", draft=draft)
                return note
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    calls_before = http_calls["n"]
    ok_cached, res_cached = run_manners_check([target_py], config=cfg, transport=httpx.MockTransport(parallel_handler))
    assert ok_cached is True
    assert [r.status for r in res_cached] == ["cached", "cached", "cached"]
    assert http_calls["n"] == calls_before

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    list_cached = runner.invoke(app, ["list", "app_mod.py", "--json"])
    assert list_cached.exit_code == 0
    assert [p["lock_status"] for p in json.loads(list_cached.stdout)["manners"]] == ["locked", "locked", "locked"]

    # Changing a function signature / docstring marks par.p1 as source_changed and triggers re-calibration.
    target_py.write_text(
        target_py.read_text(encoding="utf-8").replace('"""Greet the learner."""', '"""Greet the learner warmly."""'),
        encoding="utf-8",
    )
    list_changed = runner.invoke(app, ["list", "app_mod.py", "--json"])
    assert list_changed.exit_code == 0
    assert [p["lock_status"] for p in json.loads(list_changed.stdout)["manners"]] == [
        "source_changed",
        "locked",
        "locked",
    ]


def test_multi_case_manual_fixtures_backtest_and_hash_invalidation(tmp_path: Path) -> None:
    target_py = tmp_path / "tutor.py"
    target_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners(["Greets warmly", "Uses Spanish phrase"], id="tutor.multi")
            def greet(name: str) -> str:
                return f"Hola {name}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    fixtures_dir = tmp_path / ".manners" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    fixture_file = fixtures_dir / "tutor.multi.json"
    fixture_file.write_text(
        json.dumps(
            {
                "check_id": "tutor.multi",
                "pass": [
                    {"inputs": {"name": "Ana"}, "output": "GOOD Hola Ana!"},
                    {"inputs": {"name": "Luis"}, "output": "GOOD Buenos dias Luis!"},
                ],
                "fail": [
                    {"inputs": {"name": "Ana"}, "output": "BAD hello"},
                    {"inputs": {"name": "Luis"}, "output": "BAD hi"},
                    {"inputs": {"name": "Sol"}, "output": "BAD hey"},
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )

    transport = httpx.MockTransport(_stage3_mock_handler)
    ok, results = run_manners_check([target_py], config=cfg, transport=transport)
    assert ok is True
    assert results[0].status == "passed"
    assert len(results[0].pass_scores) == 4  # 2 pass cases x 2 checks
    assert len(results[0].fail_scores) == 6  # 3 fail cases x 2 checks
    assert {entry["case"] for entry in results[0].pass_scores} == {0, 1}
    assert {entry["case"] for entry in results[0].fail_scores} == {0, 1, 2}

    # Unchanged fixture hits cache.
    ok_cached, res_cached = run_manners_check([target_py], config=cfg, transport=transport)
    assert ok_cached is True
    assert res_cached[0].status == "cached"

    # Appending a failing case[1] inside pass list invalidates cache and pinpoints case(s) [1].
    fixture_file.write_text(
        json.dumps(
            {
                "check_id": "tutor.multi",
                "pass": [
                    {"inputs": {"name": "Ana"}, "output": "GOOD Hola Ana!"},
                    {"inputs": {"name": "Luis"}, "output": "BAD regressed output"},
                ],
                "fail": [
                    {"inputs": {"name": "Ana"}, "output": "BAD hello"},
                    {"inputs": {"name": "Luis"}, "output": "BAD hi"},
                    {"inputs": {"name": "Sol"}, "output": "BAD hey"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ok_bad_pass, res_bad_pass = run_manners_check([target_py], config=cfg, transport=transport)
    assert ok_bad_pass is False
    assert res_bad_pass[0].gate == "gate3"
    assert "case(s) [1]" in res_bad_pass[0].reason

    # Passing all pass cases while fail[2] accidentally passes pinpoints case(s) [2].
    fixture_file.write_text(
        json.dumps(
            {
                "check_id": "tutor.multi",
                "pass": [
                    {"inputs": {"name": "Ana"}, "output": "GOOD Hola Ana!"},
                    {"inputs": {"name": "Luis"}, "output": "GOOD Buenos dias Luis!"},
                ],
                "fail": [
                    {"inputs": {"name": "Ana"}, "output": "BAD hello"},
                    {"inputs": {"name": "Luis"}, "output": "BAD hi"},
                    {"inputs": {"name": "Sol"}, "output": "GOOD unexpected pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ok_bad_fail, res_bad_fail = run_manners_check([target_py], config=cfg, transport=transport)
    assert ok_bad_fail is False
    assert res_bad_fail[0].gate == "gate3"
    assert "case(s) [2]" in res_bad_fail[0].reason

    # Empty lists are rejected cleanly.
    fixture_file.write_text(
        json.dumps(
            {
                "check_id": "tutor.multi",
                "pass": [],
                "fail": [{"inputs": {"name": "Ana"}, "output": "BAD"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BackendError, match="at least one snapshot"):
        _load_manual_fixture(fixtures_dir, "tutor.multi")


def test_v05_here_rich_excerpt_and_manners_fixture_cli(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "fixture-app"
            version = "0.1.0"

            [tool.manners.models.decision]
            provider = "http_decision"
            url = "http://mock.local/decision"
            model = "jev"

            [tool.manners.models.text]
            provider = "openai_compat"
            url = "http://mock.local/v1"
            model = "qwen3.5:9b"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    app_py = tmp_path / "teach.py"
    app_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            def build_teach_turn(user_msg: str) -> str:
                \"\"\"Assemble a beginner lesson with coupled warmup flags.\"\"\"
                also_introduce = "gracias"
                warmup_recall_only = True
                target_phrase = "hola"
                mode = "teach"
                recast_text = ""
                hint = "greet"
                # padding setup lines to exceed 6-line old window
                step1 = user_msg.strip()
                step2 = step1.lower()
                step3 = f"{target_phrase} {step2}"
                reply = f"GOOD {step3}"
                manners.here(
                    "Reply aligns with target_phrase and coupled flags",
                    id="tutor.teach_turn",
                    kind=manners.type.text,
                    also_introduce=also_introduce,
                    warmup_recall_only=warmup_recall_only,
                    target_phrase=target_phrase,
                    mode=mode,
                    recast_text=recast_text,
                    hint=hint,
                    reply=reply,
                )
                return reply

            from typing import Any

            @manners("Recorded items stay coherent", id="tutor.typed_dec")
            def record_state(items: list[str], meta: dict[str, Any], count: int) -> None:
                return None
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    items = discover_manners([app_py], root=tmp_path)
    assert len(items) == 2
    item = next(p for p in items if p.check_id == "tutor.teach_turn")
    assert "Assemble a beginner lesson with coupled warmup flags." in item.source_excerpt
    assert 'also_introduce = "gracias"' in item.source_excerpt
    assert "warmup_recall_only = True" in item.source_excerpt
    orig_source_hash = item.source_hash

    # Unrelated body refactor preserves manners.here source_hash.
    app_py.write_text(
        app_py.read_text(encoding="utf-8").replace(
            "# padding setup lines to exceed 6-line old window",
            "# logging trace added here",
        ),
        encoding="utf-8",
    )
    items_after = discover_manners([app_py], root=tmp_path)
    assert next(p for p in items_after if p.check_id == "tutor.teach_turn").source_hash == orig_source_hash

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # --init scaffolds type-aware starter fixtures matching all 7 watched keys and typed @manners params/None return.
    init_res = runner.invoke(app, ["fixture", "tutor.teach_turn", "--init", "--json"])
    assert init_res.exit_code == 0, init_res.stderr
    init_data = json.loads(init_res.stdout)
    assert init_data["pass_count"] == 1
    assert init_data["fail_count"] == 1
    fixture_fp = tmp_path / ".manners" / "fixtures" / "tutor.teach_turn.json"
    assert fixture_fp.is_file()
    fixture_json = json.loads(fixture_fp.read_text(encoding="utf-8"))
    assert set(fixture_json["pass"][0]["watched"].keys()) == set(item.watched_keys)
    assert fixture_json["pass"][0]["watched"]["warmup_recall_only"] is True
    assert fixture_json["fail"][0]["watched"]["warmup_recall_only"] is False

    dec_init = runner.invoke(app, ["fixture", "tutor.typed_dec", "--init", "--json"])
    assert dec_init.exit_code == 0, dec_init.stderr
    dec_fp = tmp_path / ".manners" / "fixtures" / "tutor.typed_dec.json"
    dec_json = json.loads(dec_fp.read_text(encoding="utf-8"))
    assert dec_json["pass"][0]["inputs"] == {
        "items": ["<PASS:items>"],
        "meta": {"<PASS:meta>": "..."},
        "count": 1,
    }
    assert dec_json["fail"][0]["inputs"] == {
        "items": ["<FAIL:items>"],
        "meta": {"<FAIL:meta>": "..."},
        "count": 0,
    }
    assert dec_json["pass"][0]["output"] is None
    assert dec_json["fail"][0]["output"] is None
    dec_fp.unlink()

    # Drop the extra typed decorator check before running fixture promotion + Gate 3 check on tutor.teach_turn.
    app_py.write_text(
        app_py.read_text(encoding="utf-8").split("\nfrom typing import Any")[0].rstrip() + "\n",
        encoding="utf-8",
    )

    # Clear fixture and verify `manners fixture` promotes recorded pass + fail snapshots across runs.
    fixture_fp.unlink()
    runs_dir = tmp_path / ".manners" / "runs"
    store = MannersStore(runs_dir, create=True)
    try:
        r_fail = "01FIXTURE00000000000000001"
        store.create_run(
            r_fail,
            started_at="2026-11-18T10:00:00.000000Z",
            cmd="pytest",
            pid=101,
            decision_model="http_decision:jev",
            text_model="openai_compat:qwen3.5:9b",
        )
        store.insert_event(
            run_id=r_fail,
            call_seq=1,
            elapsed_ms=10.0,
            started_at="2026-11-18T10:00:00.010000Z",
            check_id="tutor.teach_turn",
            site="here",
            check_type="text",
            file_line="teach.py:16",
            checks=item.checks,
            payload_json=json.dumps({"watched": {k: "BAD" for k in item.watched_keys}}),
            check_scores=(CheckScore(item.checks[0], 0.08, False),),
            verdict="fail",
        )
        store.finish_run(r_fail, status="halted", finished_at="2026-11-18T10:00:01.000000Z", halt_call_seq=1)

        # Fail-only run without --init fails cleanly with guidance instead of writing half-empty fixture.
        incomplete_res = runner.invoke(app, ["fixture", "tutor.teach_turn"])
        assert incomplete_res.exit_code == 1
        assert "found 0 pass, 1 fail" in incomplete_res.stderr
        assert not fixture_fp.exists()

        # Second run records a passing snapshot -> `manners fixture` promotes both into pass + fail lists.
        r_pass = "01FIXTURE00000000000000002"
        store.create_run(
            r_pass,
            started_at="2026-11-18T10:05:00.000000Z",
            cmd="pytest",
            pid=102,
            decision_model="http_decision:jev",
            text_model="openai_compat:qwen3.5:9b",
        )
        store.insert_event(
            run_id=r_pass,
            call_seq=1,
            elapsed_ms=12.0,
            started_at="2026-11-18T10:05:00.012000Z",
            check_id="tutor.teach_turn",
            site="here",
            check_type="text",
            file_line="teach.py:16",
            checks=item.checks,
            payload_json=json.dumps({"watched": {k: "GOOD ok" for k in item.watched_keys}}),
            check_scores=(CheckScore(item.checks[0], 0.96, True),),
            verdict="pass",
        )
        store.finish_run(r_pass, status="completed", finished_at="2026-11-18T10:05:01.000000Z")
    finally:
        store.close()

    promo_res = runner.invoke(app, ["fixture", "tutor.teach_turn"])
    assert promo_res.exit_code == 0, promo_res.stderr
    assert "(1 pass, 1 fail)" in promo_res.stdout

    # Repeat invocation deduplicates identically.
    promo_again = runner.invoke(app, ["fixture", "tutor.teach_turn", "--json"])
    assert promo_again.exit_code == 0
    assert json.loads(promo_again.stdout)["pass_count"] == 1
    assert json.loads(promo_again.stdout)["fail_count"] == 1

    # Promoted fixture passes `manners check` Gate 3 directly without synthetic synthesis.
    synth_called = {"n": 0}

    def check_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if url.endswith("/chat/completions"):
            if "passing_snapshot" in body["messages"][0]["content"]:
                synth_called["n"] += 1
            return httpx.Response(200, json={"choices": [{"message": {"content": "Factual check audit."}}]})
        if body["check_id"].endswith("#gate2"):
            return httpx.Response(200, json={"probabilities": [0.05 for _ in body["checks"]]})
        ok = "GOOD" in body["snapshot"]
        return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.10 for _ in body["checks"]]})

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(runs_dir),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok_check, res_check = run_manners_check([app_py], config=cfg, transport=httpx.MockTransport(check_handler))
    assert ok_check is True
    assert res_check[0].status == "passed"
    assert synth_called["n"] == 0


def test_v051_gate2_retry_recovers_transient_disagreement(tmp_path: Path) -> None:
    target_py = tmp_path / "grade.py"
    target_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            def cycle_grade(progress_note: str) -> str:
                manners.here(
                    "When progress_note is non-empty, its summary accurately reflects learner turns",
                    id="tutor.cycle_grade",
                    kind=manners.type.text,
                    progress_note=progress_note,
                )
                return progress_note
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    gate2_temps: list[float] = []

    def retry_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if url.endswith("/chat/completions"):
            sys_msg = body["messages"][0]["content"]
            if "Gate 2" in body["messages"][0]["content"] or "audit @manners" in sys_msg:
                gate2_temps.append(body["temperature"])
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "Check [0] tests semantic summary faithfulness."}}]},
                )
            if "passing_snapshot" in sys_msg:
                payload = {
                    "notes": "ok",
                    "passing_snapshot": {"watched": {"progress_note": "GOOD accurate note"}},
                    "failing_snapshot": {"watched": {"progress_note": "BAD hallucinated note"}},
                }
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "Evidence summary."}}]})
        if body["check_id"].endswith("#gate2"):
            # Attempt 1 fails transiently (0.62 >= 0.50); attempt 2 passes (0.05 < 0.50).
            if len(gate2_temps) == 1:
                return httpx.Response(200, json={"probabilities": [0.62, 0.05, 0.05]})
            return httpx.Response(200, json={"probabilities": [0.05, 0.05, 0.05]})
        ok = "GOOD" in body["snapshot"]
        return httpx.Response(200, json={"probabilities": [0.95 if ok else 0.10]})

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok, results = run_manners_check([target_py], config=cfg, transport=httpx.MockTransport(retry_handler))
    assert ok is True
    assert results[0].status == "passed"
    assert gate2_temps == [0.0, 0.2]

    # Persistent failure across both attempts still rejects at Gate 2.
    def always_reject(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "Mechanical length check."}}]})
        return httpx.Response(200, json={"probabilities": [0.95, 0.05, 0.05]})

    ok_fail, res_fail = run_manners_check(
        [target_py],
        force=True,
        config=cfg,
        transport=httpx.MockTransport(always_reject),
    )
    assert ok_fail is False
    assert res_fail[0].gate == "gate2"


def test_v06_decorator_ignore_gate1_gate2_gate3_and_margins(tmp_path: Path, monkeypatch) -> None:
    # 1. Gate 1 rejects unknown/empty/dynamic `ignore` on @manners.
    bad_py = tmp_path / "bad_ignore.py"
    bad_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("Output summarizes metrics", id="bad.unknown", ignore=("missing_param",))
            def fn_a(account: str) -> str:
                return account

            @manners("Output summarizes metrics", id="bad.empty", ignore="")
            def fn_b(account: str) -> str:
                return account

            DYN = ("x",)
            @manners("Output summarizes metrics", id="bad.dyn", ignore=DYN)
            def fn_c(account: str) -> str:
                return account
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok_bad, res_bad = run_manners_check([bad_py], config=cfg)
    assert ok_bad is False
    by_id = {r.check_id: r for r in res_bad}
    assert by_id["bad.unknown"].gate == "gate1" and "unknown parameter" in by_id["bad.unknown"].reason
    assert by_id["bad.empty"].gate == "gate1" and "empty" in by_id["bad.empty"].reason
    assert by_id["bad.dyn"].gate == "gate1" and "literal" in by_id["bad.dyn"].reason

    # 2. Valid @manners(..., ignore=...) filters params, prompts, --init skeletons, and surfaces margins.
    good_py = tmp_path / "svc.py"
    good_py.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            from manners import manners

            @manners("Output summarizes portfolio metrics cleanly", id="svc.summary", ignore=("data_dir", "llm_client"))
            def run_report(account: str, data_dir: Path, llm_client: object) -> str:
                return account
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    bad_py.unlink()

    seen_text_users: list[str] = []

    def ok_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content.decode("utf-8"))
        if url.endswith("/chat/completions"):
            sys_msg = body["messages"][0]["content"]
            user_msg = body["messages"][1]["content"]
            seen_text_users.append(user_msg)
            if "audit @manners" in sys_msg:
                return httpx.Response(200, json={"choices": [{"message": {"content": "Check [0] checks output."}}]})
            payload = {
                "notes": "ok",
                "passing_snapshot": {"inputs": {"account": "acct-1"}, "output": "GOOD portfolio summary."},
                "failing_snapshot": {"inputs": {"account": "acct-1"}, "output": "BAD garbled text."},
            }
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
        if body["check_id"].endswith("#gate2"):
            return httpx.Response(200, json={"probabilities": [0.04, 0.04, 0.04]})
        ok = "GOOD" in body["snapshot"]
        return httpx.Response(200, json={"probabilities": [0.93 if ok else 0.12]})

    ok_good, res_good = run_manners_check(
        [good_py], config=cfg, transport=httpx.MockTransport(ok_handler)
    )
    assert ok_good is True
    assert len(res_good) == 1
    assert "Passed Gate 2 audit & Gate 3 synthetic calibration" in res_good[0].reason
    assert "(pass min p=0.93, fail max p=0.12)" in res_good[0].reason
    assert len(seen_text_users) == 2
    expected_hdr = "Captured inputs: account (ignored parameters omitted at runtime: data_dir, llm_client)"
    assert expected_hdr in seen_text_users[0]
    assert expected_hdr in seen_text_users[1]

    lock_data = json.loads(cfg.resolve_lock_path().read_text(encoding="utf-8"))
    entry = lock_data["manners"]["svc.summary"]
    assert entry["param_names"] == ["account"]
    assert entry["ignored_params"] == ["data_dir", "llm_client"]
    assert entry["pass_min_p"] == pytest.approx(0.93)
    assert entry["fail_max_p"] == pytest.approx(0.12)

    # Cached run also surfaces stored calibration margins.
    ok_cached, res_cached = run_manners_check(
        [good_py], config=cfg, transport=httpx.MockTransport(ok_handler)
    )
    assert ok_cached is True
    assert res_cached[0].status == "cached"
    assert "pass min p=0.93, fail max p=0.12" in res_cached[0].reason

    # Snapshot validation rejects accidental ignored keys.
    disc_item = discover_manners([good_py], root=tmp_path)[0]
    with pytest.raises(BackendError, match="must omit ignored input"):
        _validate_gate3_snapshot(
            json.dumps({"inputs": {"account": "acct-1", "data_dir": "/tmp"}, "output": "ok"}),
            item=disc_item,
            label="passing_snapshot",
        )

    # Fixture --init omits ignored parameters.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='ign'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    fix_res = CliRunner().invoke(app, ["fixture", "svc.summary", "--init", "--json"])
    assert fix_res.exit_code == 0, fix_res.stdout
    fix_payload = json.loads((tmp_path / ".manners" / "fixtures" / "svc.summary.json").read_text(encoding="utf-8"))
    assert fix_payload["pass"] == [{"inputs": {"account": "<PASS:account>"}, "output": "<PASS:output>"}]
    assert fix_payload["fail"] == [{"inputs": {"account": "<FAIL:account>"}, "output": "<FAIL:output>"}]


def test_v06_cli_root_and_monorepo_check_list_fixture_report(tmp_path: Path, monkeypatch) -> None:
    mono = tmp_path / "monorepo"
    (mono / ".git").mkdir(parents=True)
    sub = mono / "services" / "portfolio"
    pkg_dir = sub / "src" / "portfolio"
    pkg_dir.mkdir(parents=True)

    # Local loopback mock server for full CLI `manners check --root services/portfolio` verification.
    class MockHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path.endswith("/chat/completions"):
                sys_msg = body["messages"][0]["content"]
                if "audit @manners" in sys_msg:
                    resp = {"choices": [{"message": {"content": "Check [0] evaluates reply quality."}}]}
                else:
                    snap = {
                        "notes": "ok",
                        "passing_snapshot": {"inputs": {"ticker": "AAPL"}, "output": "GOOD reply."},
                        "failing_snapshot": {"inputs": {"ticker": "AAPL"}, "output": "BAD reply."},
                    }
                    resp = {"choices": [{"message": {"content": json.dumps(snap)}}]}
            else:
                if body.get("check_id", "").endswith("#gate2"):
                    resp = {"probabilities": [0.03, 0.03, 0.03]}
                else:
                    ok = "GOOD" in body.get("snapshot", "")
                    resp = {"probabilities": [0.91 if ok else 0.08]}
            raw = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        (mono / ".env").write_text("DEEPSEEK_API_KEY=mono-root-key-123\n", encoding="utf-8")
        (sub / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [project]
                name = "portfolio"
                [tool.manners.models.decision]
                provider = "http_decision"
                url = "{base_url}/decision"
                model = "local-decision"
                [tool.manners.models.text]
                provider = "openai_compat"
                url = "{base_url}/v1"
                model = "qwen3.5:9b"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        (pkg_dir / "app.py").write_text(
            textwrap.dedent(
                """
                from manners import manners

                @manners("Reply states clear metric summary", id="portfolio.app.summary")
                def summarize(ticker: str) -> str:
                    return f"{ticker}: ok"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        # Populate two runs in services/portfolio/.manners/runs for report/log/runs/clean.
        store = MannersStore(sub / ".manners" / "runs", create=True)
        try:
            store.create_run(
                "01RUNA00000000000000000001",
                started_at="2026-11-18T10:00:00Z",
                cmd="pytest old",
                pid=1,
                decision_model="d",
                text_model="t",
            )
            store.finish_run("01RUNA00000000000000000001", status="completed", finished_at="2026-11-18T10:00:01Z")
            store.create_run(
                "01RUNB00000000000000000002",
                started_at="2026-11-18T11:00:00Z",
                cmd="pytest new",
                pid=2,
                decision_model="d",
                text_model="t",
            )
            store.insert_event(
                run_id="01RUNB00000000000000000002",
                call_seq=1,
                elapsed_ms=10.0,
                started_at="2026-11-18T11:00:00.010Z",
                check_id="portfolio.app.summary",
                site="decorator",
                check_type="decision",
                file_line="src/portfolio/app.py:4",
                checks=("Reply states clear metric summary",),
                payload_json='{"inputs": {"ticker": "AAPL"}, "output": "AAPL: ok"}',
                check_scores=(CheckScore("Reply states clear metric summary", 0.92, True),),
                verdict="pass",
            )
            store.finish_run("01RUNB00000000000000000002", status="completed", finished_at="2026-11-18T11:00:01Z")
        finally:
            store.close()

        monkeypatch.setenv("MANNERS_CONFIG", str(tmp_path / "no_user.toml"))
        monkeypatch.delenv("MANNERS_PROJECT_ROOT", raising=False)
        monkeypatch.chdir(mono)
        runner = CliRunner()

        list_global = runner.invoke(app, ["--root", "services/portfolio", "list", "--json"])
        assert list_global.exit_code == 0, list_global.stdout
        assert [p["check_id"] for p in json.loads(list_global.stdout)["manners"]] == ["portfolio.app.summary"]

        chk = runner.invoke(app, ["check", "--root", "services/portfolio"])
        assert chk.exit_code == 0, chk.stdout + chk.stderr
        assert "[PASSED] portfolio.app.summary (src/portfolio/app.py:4)" in chk.stdout
        assert "Passed Gate 2 audit & Gate 3 synthetic calibration (pass min p=0.91, fail max p=0.08)" in chk.stdout
        assert (sub / ".manners" / "lock.json").is_file()
        assert not (mono / ".manners").exists()

        fix = runner.invoke(app, ["fixture", "portfolio.app.summary", "--init", "--root", "services/portfolio"])
        assert fix.exit_code == 0, fix.stdout
        assert (sub / ".manners" / "fixtures" / "portfolio.app.summary.json").is_file()

        rep = runner.invoke(app, ["report", "--all", "--root", "services/portfolio"])
        assert rep.exit_code == 0 and "portfolio.app.summary" in rep.stdout

        log_res = runner.invoke(app, ["log", "portfolio.*", "--root", "services/portfolio"])
        assert log_res.exit_code == 0 and "portfolio.app.summary" in log_res.stdout

        runs_res = runner.invoke(app, ["runs", "--json", "--root", "services/portfolio"])
        assert runs_res.exit_code == 0 and len(json.loads(runs_res.stdout)["runs"]) == 2

        clean_res = runner.invoke(app, ["clean", "--keep", "1", "--root", "services/portfolio"])
        assert clean_res.exit_code == 0 and "Removed 1" in clean_res.stdout
    finally:
        server.shutdown()
        server.server_close()


def test_v061_gate3_truncation_retry_max_items_50_and_import_hint(tmp_path: Path) -> None:
    from manners.redact import SecretRedactor, serialize_call_snapshot

    # 1. Gate 3 omits max_tokens by default and retries once with concise-output feedback on finish_reason='length'.
    app_py = tmp_path / "summary.py"
    app_py.write_text(
        textwrap.dedent(
            """
            from manners import manners

            @manners("Summary captures all key action items", id="app.summary")
            def make_summary(notes: str) -> dict[str, str]:
                return {"summary": notes}
            """
        ).strip()
        + "\n",
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
                    return httpx.Response(
                        200,
                        json={
                            "choices": [
                                {
                                    "finish_reason": "length",
                                    "message": {"content": '{"notes": "truncated...'},
                                }
                            ]
                        },
                    )
                payload = {
                    "notes": "concise",
                    "passing_snapshot": {"inputs": {"notes": "ok"}, "output": {"summary": "GOOD summary"}},
                    "failing_snapshot": {"inputs": {"notes": "ok"}, "output": {"summary": "BAD empty"}},
                }
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Check evaluates summary completeness."}}]},
            )
        if body["check_id"].endswith("#gate2"):
            return httpx.Response(200, json={"probabilities": [0.05 for _ in body["checks"]]})
        ok = "GOOD" in body["snapshot"]
        return httpx.Response(200, json={"probabilities": [0.94 if ok else 0.08]})

    cfg = MannersConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="jev"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        decision=DecisionConfig(pass_threshold=0.50),
        text=TextConfig(),
        check=CheckConfig(require_check=True),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".manners" / "runs"),
            lock_path=str(tmp_path / ".manners" / "lock.json"),
        ),
        project_root=tmp_path,
    )
    ok_check, results = run_manners_check([app_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok_check is True
    assert results[0].status == "passed"
    assert len(gate3_calls) == 2
    assert "max_tokens" not in gate3_calls[0]
    assert "max_tokens" not in gate3_calls[1]
    assert "finish_reason='length'" in gate3_calls[1]["messages"][1]["content"]

    # 2. Default max_items is 50 (a 30-field return dict stays intact; 55-field dict truncates 5).
    assert RuntimeConfig().max_items == 50
    redactor = SecretRedactor()
    snap_30 = json.loads(
        serialize_call_snapshot(
            {"x": 1},
            redactor=redactor,
            return_value={f"f{i}": i for i in range(30)},
            has_return=True,
        )
    )
    assert "__truncated_items__" not in snap_30["output"]
    assert len(snap_30["output"]) == 30

    snap_55 = json.loads(
        serialize_call_snapshot(
            {"x": 1},
            redactor=redactor,
            return_value={f"f{i}": i for i in range(55)},
            has_return=True,
        )
    )
    assert snap_55["output"]["__truncated_items__"] == 5

    # 3. Gate 1 import hint explicitly mentions `from manners import manners` and `manners.type`.
    bad_import_py = tmp_path / "bad_import.py"
    bad_import_py.write_text(
        textwrap.dedent(
            """
            import manners

            @manners("Output summarizes metrics", id="app.bad")
            def bad(x: str) -> str:
                return x
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    ok_bad, res_bad = run_manners_check([bad_import_py], config=cfg, transport=httpx.MockTransport(handler))
    assert ok_bad is False
    assert res_bad[0].gate == "gate1"
    assert "from manners import manners" in res_bad[0].reason
    assert "manners.type" in res_bad[0].reason
