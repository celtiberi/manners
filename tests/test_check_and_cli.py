"""Tests for AST discovery, probe check Gates 1-3, lockfile caching, and all CLI commands."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import httpx
from typer.testing import CliRunner

from probe.backends import CheckScore
from probe.check import discover_probes, run_probe_check
from probe.cli import app
from probe.config import (
    CheckConfig,
    DecisionConfig,
    ModelEndpointConfig,
    ProbeConfig,
    RuntimeConfig,
    TextConfig,
)
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
                                            "passing_snapshot": {"output": "GOOD Hola, say you want coffee"},
                                            "failing_snapshot": {"output": "BAD Quiero un cafe"},
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
            db_path=str(tmp_path / ".probes" / "probes.db"),
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
            db_path=str(tmp_path / ".probes" / "probes.db"),
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

    db_path = tmp_path / ".probes" / "probes.db"
    store = ProbeStore(db_path, create=True)
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
            when_cond=None,
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
            when_cond="draft cites external facts",
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
            when_cond=None,
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

    runs_res = runner.invoke(app, ["runs", "--json"])
    assert runs_res.exit_code == 0
    assert len(json.loads(runs_res.stdout)["runs"]) == 2

    clean_res = runner.invoke(app, ["clean", "--keep", "1"])
    assert clean_res.exit_code == 0
    assert "Removed 1 old run(s)" in clean_res.stdout
    runs_after = runner.invoke(app, ["runs", "--json"])
    remaining = json.loads(runs_after.stdout)["runs"]
    assert len(remaining) == 1 and remaining[0]["id"] == run_id
