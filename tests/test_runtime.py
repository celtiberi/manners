"""Tests for @probe, probe.here, ordering, text/decision pipelines, sync halts, and rate limits."""

from __future__ import annotations

import asyncio
import json
import pdb
import time
from pathlib import Path

import httpx
import pytest

from probe import ProbeHaltError, probe, reset_for_testing
from probe.config import (
    CheckConfig,
    DecisionConfig,
    ModelEndpointConfig,
    ProbeConfig,
    RateLimitConfig,
    RuntimeConfig,
    TextConfig,
)
from probe.store import ProbeStore


def _make_config(
    tmp_path: Path,
    *,
    require_check: bool = False,
    max_calls: int = 20,
    window_seconds: float = 10.0,
    explain_failures: bool = True,
) -> ProbeConfig:
    return ProbeConfig(
        models_decision=ModelEndpointConfig(
            provider="http_decision",
            url="http://mock.local/decision",
            model="local-decision",
        ),
        models_text=ModelEndpointConfig(
            provider="openai_compat",
            url="http://mock.local/v1",
            model="qwen3.5:9b",
        ),
        decision=DecisionConfig(pass_threshold=0.50),
        text=TextConfig(detail="standard", max_tokens=300, explain_decision_failures=explain_failures),
        rate_limit=RateLimitConfig(max_calls=max_calls, window_seconds=window_seconds, max_calls_per_run=100),
        check=CheckConfig(require_check=require_check),
        runtime=RuntimeConfig(
            db_path=str(tmp_path / ".probes" / "probes.db"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
            max_concurrency=4,
            drain_timeout_s=5.0,
        ),
        project_root=tmp_path,
    )


def test_disabled_probes_returns_unwrapped_function_and_noop_here(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PROBES", raising=False)
    reset_for_testing(config=_make_config(tmp_path))

    def raw_sync(x: int) -> int:
        probe.here("should do nothing", val=x)
        return x + 1

    decorated = probe("check output", id="noop.sync")(raw_sync)
    assert decorated is raw_sync
    assert decorated(2) == 3

    async def raw_async(x: int) -> int:
        return x * 2

    decorated_async = probe("check async", id="noop.async")(raw_async)
    assert decorated_async is raw_async
    assert not (tmp_path / ".probes" / "probes.db").exists()


def test_async_call_order_preserved_under_out_of_order_completion_and_entry_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBES", "1")
    seen_snapshots: dict[str, dict] = {}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if str(request.url).endswith("/decision"):
            snap = json.loads(body["snapshot"])
            pid = body["probe_id"]
            seen_snapshots[f"{pid}:{snap}"] = snap
            # Force call #1 (alpha) to finish slower than call #2 (beta) and #3 (here).
            if pid == "step.alpha":
                time.sleep(0.08)
            return httpx.Response(200, json={"probabilities": [0.95] * len(body["checks"])})
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Grounded strictly in chunks."}}]},
            )
        raise AssertionError(f"Unexpected URL {request.url}")

    cfg = _make_config(tmp_path)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(transport_handler))

    @probe("Alpha mutates input safely after snapshot", id="step.alpha")
    def fn_alpha(items: list[str]) -> str:
        items.append("mutated_inside_fn")
        return "done"

    @probe("Beta runs second asynchronously", id="step.beta", kind=probe.type.text)
    async def fn_beta(prompt: str) -> str:
        await asyncio.sleep(0.005)
        probe.here(
            "Watchpoint draft grounded",
            id="rag.watch",
            kind=probe.type.decision,
            draft="Answer based on source",
        )
        return "beta_done"

    try:
        arg_list = ["initial"]
        assert fn_alpha(arg_list) == "done"
        assert asyncio.run(fn_beta("question")) == "beta_done"
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()

    assert rt is not None
    store = ProbeStore(cfg.resolve_db_path(), create=False)
    try:
        events = store.list_events(rt.run_id)
    finally:
        store.close()

    # call_seq 1 = alpha (entry snapshot captured before mutation), 2 = beta, 3 = rag.watch
    assert [(e.call_seq, e.probe_id, e.verdict) for e in events] == [
        (1, "step.alpha", "pass"),
        (2, "step.beta", "pass"),
        (3, "rag.watch", "pass"),
    ]
    alpha_payload = json.loads(events[0].payload_json)
    assert alpha_payload["inputs"]["items"] == ["initial"]
    assert alpha_payload["output"] == "done"
    assert events[1].text_analysis == "Grounded strictly in chunks."


def test_sync_true_halt_non_tty_and_mock_tty_pdb(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "0")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if str(request.url).endswith("/decision"):
            n = len(body["checks"])
            probs = [0.94, 0.12] if n == 2 else [0.12] * n
            return httpx.Response(200, json={"probabilities": probs})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Tutor spoiled the target Spanish sentence."}}]},
        )

    cfg = _make_config(tmp_path)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))

    @probe(
        [
            "Opens with Spanish greeting",
            "Does NOT give away target Spanish sentence",
        ],
        id="tutor.opening_turn",
        sync=True,
    )
    def generate_turn(student: str) -> str:
        return "Try saying: Quiero un café por favor"

    with pytest.raises(ProbeHaltError) as exc_info:
        generate_turn("Alex")
    assert "tutor.opening_turn" in str(exc_info.value)

    rt = probe.flush_for_testing()
    reset_for_testing()
    assert rt is not None
    store = ProbeStore(cfg.resolve_db_path(), create=False)
    try:
        run = store.get_run(rt.run_id)
        events = store.list_events(rt.run_id)
    finally:
        store.close()
    assert run is not None and run.status == "halted" and run.halt_call_seq == 1
    assert events[0].verdict == "fail"
    assert events[0].text_analysis == "Tutor spoiled the target Spanish sentence."

    # Now verify TTY mode calls pdb.Pdb.set_trace with caller frame.
    pdb_frames: list[object] = []
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(pdb.Pdb, "set_trace", lambda self, frame=None: pdb_frames.append(frame))

    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))

    def outer_caller() -> None:
        probe.here("Must pass", id="tty.here", sync=True, x=1)

    outer_caller()
    probe.flush_for_testing()
    reset_for_testing()
    assert len(pdb_frames) == 1
    assert pdb_frames[0].f_code.co_name == "outer_caller"  # type: ignore[attr-defined]


def test_sync_when_condition_triggers_only_when_condition_holds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        snap = json.loads(body["snapshot"])["watched"]
        if body["probe_id"].endswith("#sync_when"):
            triggered = snap["draft"].startswith("BAD")
            return httpx.Response(200, json={"probabilities": [0.95 if triggered else 0.05]})
        return httpx.Response(200, json={"probabilities": [0.92]})

    cfg = _make_config(tmp_path)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))

    try:
        # Condition does not hold -> continues cleanly.
        probe.here(
            "Grounded draft",
            id="rag.draft_answer",
            sync="when draft looks ungrounded",
            draft="GOOD answer",
        )
        # Condition holds -> halts cleanly inline.
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here(
                "Grounded draft",
                id="rag.draft_answer",
                sync="when draft looks ungrounded",
                draft="BAD answer",
            )
        assert "Sync breakpoint triggered" in str(exc_info.value)
    finally:
        probe.flush_for_testing()
        reset_for_testing()


def test_async_failure_triggers_interrupt_callback_and_subsequent_halt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROBES", "1")
    interrupted: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/decision"):
            return httpx.Response(200, json={"probabilities": [0.05]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Failed async."}}]})

    cfg = _make_config(tmp_path)
    reset_for_testing(
        config=cfg,
        transport=httpx.MockTransport(handler),
        interrupt_main_fn=lambda: interrupted.append(True),
    )

    try:
        probe.here("Should fail async", id="async.fail", x=10)
        deadline = time.monotonic() + 2.0
        while not interrupted and time.monotonic() < deadline:
            time.sleep(0.02)
        assert interrupted == [True]

        with pytest.raises(ProbeHaltError):
            probe.here("Next call stops", id="async.next", x=11)
    finally:
        probe.flush_for_testing()
        reset_for_testing()


def test_rate_limit_trips_probe_misuse_and_lock_enforcement(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROBES", "1")
    model_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        model_calls["n"] += 1
        return httpx.Response(200, json={"probabilities": [0.9]})

    # 1. Tight loop exceeds max_calls=2 within 10s -> trips PROBE_MISUSE on call #3 before model call #3.
    cfg = _make_config(tmp_path, max_calls=2, window_seconds=10.0)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))
    try:
        probe.here("Loop check", id="loop.hot", sync=True, i=0)
        probe.here("Loop check", id="loop.hot", sync=True, i=1)
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here("Loop check", id="loop.hot", sync=True, i=2)
        assert "PROBE_MISUSE" in str(exc_info.value)
        assert model_calls["n"] == 2
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()

    assert rt is not None
    store = ProbeStore(cfg.resolve_db_path(), create=False)
    try:
        events = store.list_events(rt.run_id)
    finally:
        store.close()
    assert events[-1].error_code == "PROBE_MISUSE"
    assert events[-1].stack_excerpt

    # 2. require_check=True refuses unlocked probes unless PROBE_SKIP_LOCK=1.
    locked_cfg = _make_config(tmp_path, require_check=True)
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here("Locked check", id="needs.lock", sync=True, i=0)
        assert "probe check" in str(exc_info.value)
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    monkeypatch.setenv("PROBE_SKIP_LOCK", "1")
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))
    try:
        probe.here("Locked check", id="needs.lock", sync=True, i=0)
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    # 3. Matching checks_hash in .probes/lock.json allows execution without PROBE_SKIP_LOCK.
    monkeypatch.delenv("PROBE_SKIP_LOCK", raising=False)
    from probe.runtime import ProbeType, compute_checks_hash

    lock_file = tmp_path / ".probes" / "lock.json"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(
        json.dumps(
            {
                "version": 1,
                "probes": {
                    "needs.lock": {
                        "checks_hash": compute_checks_hash(ProbeType.decision, ("Locked check",)),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))
    try:
        probe.here("Locked check", id="needs.lock", sync=True, i=0)
        with pytest.raises(ProbeHaltError):
            probe.here("Edited check not locked yet", id="needs.lock", sync=True, i=1)
    finally:
        probe.flush_for_testing()
        reset_for_testing()
