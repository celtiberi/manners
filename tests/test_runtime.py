"""Tests for synchronous @probe, probe.here, .env flags, nested ordering, pdb halts, when=, and rate limits."""

from __future__ import annotations

import asyncio
import json
import pdb
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
    is_probes_enabled,
)
from probe.runtime import ProbeType, compute_checks_hash
from probe.store import ProbeStore


def _make_config(
    tmp_path: Path,
    *,
    require_check: bool = False,
    max_calls: int = 20,
    window_seconds: float = 10.0,
    max_calls_per_run: int = -1,
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
        rate_limit=RateLimitConfig(
            max_calls=max_calls,
            window_seconds=window_seconds,
            max_calls_per_run=max_calls_per_run,
        ),
        check=CheckConfig(require_check=require_check),
        runtime=RuntimeConfig(
            db_path=str(tmp_path / ".probes" / "probes.db"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )


def test_disabled_probes_returns_unwrapped_function_and_noop_here(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PROBES", raising=False)
    monkeypatch.chdir(tmp_path)
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


def test_dotenv_probes_flag_and_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PROBES", raising=False)
    (tmp_path / ".env").write_text("PROBES=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert is_probes_enabled(tmp_path) is True

    # Shell env PROBES=0 overrides .env PROBES=1.
    monkeypatch.setenv("PROBES", "0")
    assert is_probes_enabled(tmp_path) is False


def test_nested_inline_call_entry_ordering_and_entry_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBES", "1")
    eval_order: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if str(request.url).endswith("/decision"):
            eval_order.append(body["probe_id"])
            return httpx.Response(200, json={"probabilities": [0.95] * len(body["checks"])})
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Grounded strictly in chunks."}}]},
            )
        raise AssertionError(f"Unexpected URL {request.url}")

    cfg = _make_config(tmp_path)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(transport_handler))

    @probe("Inner async step check", id="step.inner", kind=probe.type.text)
    async def fn_inner(prompt: str) -> str:
        await asyncio.sleep(0.005)
        probe.here(
            "Watchpoint draft grounded",
            id="rag.watch",
            kind=probe.type.decision,
            draft="Answer based on source",
        )
        return "inner_done"

    @probe("Outer sync step mutates list safely after snapshot", id="step.outer")
    def fn_outer(items: list[str]) -> str:
        items.append("mutated_inside_outer")
        inner_res = asyncio.run(fn_inner("question"))
        return f"outer:{inner_res}"

    try:
        arg_list = ["initial"]
        assert fn_outer(arg_list) == "outer:inner_done"
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()

    assert rt is not None
    # Execution finished inner -> here -> outer, while call_seq stays entry-ordered 1=outer, 2=inner, 3=here.
    assert eval_order == ["rag.watch", "step.inner", "step.outer"]

    store = ProbeStore(cfg.resolve_db_path(), create=False)
    try:
        events = store.list_events(rt.run_id)
    finally:
        store.close()

    assert [(e.call_seq, e.probe_id, e.verdict) for e in events] == [
        (1, "step.outer", "pass"),
        (2, "step.inner", "pass"),
        (3, "rag.watch", "pass"),
    ]
    outer_payload = json.loads(events[0].payload_json)
    assert outer_payload["inputs"]["items"] == ["initial"]
    assert outer_payload["output"] == "outer:inner_done"
    assert events[1].text_analysis == "Grounded strictly in chunks."


def test_inline_halt_non_tty_and_mock_tty_pdb(tmp_path: Path, monkeypatch) -> None:
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
    )
    def generate_turn(student: str) -> str:
        return "Try saying: Quiero un café por favor"

    with pytest.raises(ProbeHaltError) as exc_info:
        generate_turn("Alex")
    assert "tutor.opening_turn" in str(exc_info.value)

    # Subsequent calls halt immediately once tripped.
    with pytest.raises(ProbeHaltError):
        probe.here("Should not run after trip", id="after.trip", x=1)

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
    assert len(events) == 1
    assert events[0].verdict == "fail"
    assert events[0].text_analysis == "Tutor spoiled the target Spanish sentence."

    # Verify interactive TTY mode opens pdb on the caller's live stack frame.
    pdb_frames: list[object] = []
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(pdb.Pdb, "set_trace", lambda self, frame=None: pdb_frames.append(frame))

    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))

    def outer_caller() -> None:
        probe.here("Must pass", id="tty.here", x=1)

    outer_caller()
    probe.flush_for_testing()
    reset_for_testing()
    assert len(pdb_frames) == 1
    assert pdb_frames[0].f_code.co_name == "outer_caller"  # type: ignore[attr-defined]


def test_when_condition_triggers_only_when_condition_holds(tmp_path: Path, monkeypatch) -> None:
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
            when="draft cites figures absent from source",
            draft="GOOD answer",
        )
        # Condition holds -> halts cleanly inline.
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here(
                "Grounded draft",
                id="rag.draft_answer",
                when="draft cites figures absent from source",
                draft="BAD answer",
            )
        assert "Breakpoint condition matched" in str(exc_info.value)
    finally:
        probe.flush_for_testing()
        reset_for_testing()


def test_rate_limit_unlimited_per_run_default_sliding_window_and_lock(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBES", "1")
    model_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        model_calls["n"] += 1
        return httpx.Response(200, json={"probabilities": [0.9]})

    # 1. Default max_calls_per_run=-1 allows repeated calls as long as sliding window permits.
    cfg_unlimited = _make_config(tmp_path, max_calls=10, window_seconds=10.0, max_calls_per_run=-1)
    reset_for_testing(config=cfg_unlimited, transport=httpx.MockTransport(handler))
    try:
        for i in range(6):
            probe.here("Allowed server check", id="server.route", i=i)
        assert model_calls["n"] == 6
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    # 2. Explicit positive max_calls_per_run caps calls per run cleanly.
    model_calls["n"] = 0
    cfg_cap = _make_config(tmp_path, max_calls=10, window_seconds=10.0, max_calls_per_run=2)
    reset_for_testing(config=cfg_cap, transport=httpx.MockTransport(handler))
    try:
        probe.here("Capped check", id="capped.hot", i=0)
        probe.here("Capped check", id="capped.hot", i=1)
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here("Capped check", id="capped.hot", i=2)
        assert "max_calls_per_run" in str(exc_info.value)
        assert model_calls["n"] == 2
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    # 3. Tight loop exceeds max_calls=2 within 10s -> trips PROBE_MISUSE on call #3 before model call #3.
    model_calls["n"] = 0
    cfg = _make_config(tmp_path, max_calls=2, window_seconds=10.0)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))
    try:
        probe.here("Loop check", id="loop.hot", i=0)
        probe.here("Loop check", id="loop.hot", i=1)
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here("Loop check", id="loop.hot", i=2)
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

    # 4. require_check=True refuses unlocked probes unless PROBE_SKIP_LOCK=1 or lock matches.
    locked_cfg = _make_config(tmp_path, require_check=True)
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProbeHaltError) as exc_info:
            probe.here("Locked check", id="needs.lock", i=0)
        assert "probe check" in str(exc_info.value)
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    monkeypatch.setenv("PROBE_SKIP_LOCK", "1")
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))
    try:
        probe.here("Locked check", id="needs.lock", i=0)
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    monkeypatch.delenv("PROBE_SKIP_LOCK", raising=False)
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
        probe.here("Locked check", id="needs.lock", i=0)
        with pytest.raises(ProbeHaltError):
            probe.here("Edited check not locked yet", id="needs.lock", i=1)
    finally:
        probe.flush_for_testing()
        reset_for_testing()
