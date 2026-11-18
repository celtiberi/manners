"""Tests for synchronous @probe, probe.here, .env flags, nested ordering, pdb halts, and rate limits."""

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
    clear_env_cache,
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
            runs_dir=str(tmp_path / ".probes" / "runs"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
        ),
        project_root=tmp_path,
    )


def test_disabled_probes_passes_through_and_enables_dynamically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PROBES", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = _make_config(tmp_path)
    reset_for_testing(config=cfg)

    def raw_sync(x: int) -> int:
        probe.here("should do nothing", val=x)
        return x + 1

    decorated = probe("check output", id="noop.sync")(raw_sync)
    assert decorated.__wrapped__ is raw_sync
    assert decorated(2) == 3

    async def raw_async(x: int) -> int:
        return x * 2

    decorated_async = probe("check async", id="noop.async")(raw_async)
    assert decorated_async.__wrapped__ is raw_async
    assert asyncio.run(decorated_async(3)) == 6
    assert not (tmp_path / ".probes" / "runs").exists()

    # Enabling PROBES=1 after module import / decoration activates wrapped probes cleanly.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"probabilities": [0.95]})

    monkeypatch.setenv("PROBES", "1")
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))
    try:
        assert decorated(4) == 5
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()
    assert rt is not None
    store = ProbeStore(cfg.resolve_runs_dir(), create=False)
    try:
        events = store.list_events(rt.run_id)
    finally:
        store.close()
    assert len(events) == 2
    assert events[0].probe_id == "noop.sync"
    assert events[0].verdict == "pass"
    assert events[1].site == "here"
    assert events[1].verdict == "pass"


def test_dotenv_probes_flag_and_env_override(tmp_path: Path, monkeypatch) -> None:
    clear_env_cache()
    monkeypatch.delenv("PROBES", raising=False)
    (tmp_path / ".env").write_text("PROBES=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert is_probes_enabled(tmp_path) is True

    # Shell env PROBES=0 or empty PROBES="" overrides .env PROBES=1.
    monkeypatch.setenv("PROBES", "0")
    assert is_probes_enabled(tmp_path) is False
    monkeypatch.setenv("PROBES", "")
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

    store = ProbeStore(cfg.resolve_runs_dir(), create=False)
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
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

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
    store = ProbeStore(cfg.resolve_runs_dir(), create=False)
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
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
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
    store = ProbeStore(cfg.resolve_runs_dir(), create=False)
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
                        "site": "here",
                        "watched_keys": ["i"],
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


def test_exceptions_descriptors_generators_preexec_guards_and_ro_store(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    seen_model_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_model_calls.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"probabilities": [0.95]})

    cfg = _make_config(tmp_path, require_check=False, max_calls=1, window_seconds=10.0)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))

    # 1. Target Exception propagates unmodified without calling decision model (no vacuous pass).
    @probe("Output does NOT spoil secret", id="boom.fn")
    def crasher(x: int) -> str:
        raise ValueError(f"bad input {x}")

    try:
        with pytest.raises(ValueError, match="bad input 7"):
            crasher(7)
        assert seen_model_calls == []

        # 2. @probe stacked above @classmethod and @staticmethod works under PROBES=1.
        class Builder:
            @probe("Builds string", id="builder.cls")
            @classmethod
            def from_tag(cls, tag: str) -> str:
                return f"{cls.__name__}:{tag}"

            @probe("Formats item", id="builder.static")
            @staticmethod
            def fmt(item: str) -> str:
                return item.upper()

        assert Builder.from_tag("ok") == "Builder:ok"
        assert Builder().fmt("hi") == "HI"

        # 3. Generator / async generator decorated with @probe fails fast at decoration time.
        with pytest.raises(TypeError, match="generator"):
            @probe("Stream chunk")
            def gen():
                yield 1

        with pytest.raises(TypeError, match="generator"):
            @probe("Async stream chunk")
            async def agen():
                yield 1
    finally:
        rt = probe.flush_for_testing()
        reset_for_testing()

    assert rt is not None
    ro_dir = cfg.resolve_runs_dir()
    ro_run = ro_dir / f"{rt.run_id}.jsonl"
    ro_run.chmod(0o444)
    ro_dir.chmod(0o555)
    try:
        store = ProbeStore(ro_dir, create=False)
        try:
            events = store.list_events(rt.run_id)
        finally:
            store.close()
    finally:
        ro_dir.chmod(0o755)
        ro_run.chmod(0o644)

    assert events[0].error_code == "TARGET_EXCEPTION"
    assert events[0].verdict == "error"
    assert "bad input 7" in (events[0].text_analysis or "")
    assert [e.probe_id for e in events[1:]] == ["builder.cls", "builder.static"]

    # 4. Pre-execution verify_lock prevents side effects inside fn body.
    side_effects = {"ran": 0}
    locked_cfg = _make_config(tmp_path, require_check=True)
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))

    @probe("Locked check", id="unlocked.side_effect")
    def do_write() -> str:
        side_effects["ran"] += 1
        return "wrote"

    try:
        with pytest.raises(ProbeHaltError, match="lock check"):
            do_write()
        assert side_effects["ran"] == 0
    finally:
        probe.flush_for_testing()
        reset_for_testing()


def test_disabled_context_class_rejection_tty_entry_halt_and_jsonl_hardening(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "1")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"probabilities": [0.95]})

    cfg = _make_config(tmp_path)
    reset_for_testing(config=cfg, transport=httpx.MockTransport(handler))

    @probe("Friendly output", id="disabled.sync")
    def sync_fn(x: str) -> str:
        probe.here("Non-empty draft", id="disabled.here", x=x)
        return x

    @probe("Friendly async output", id="disabled.async")
    async def async_fn(x: str) -> str:
        return x

    try:
        with probe.disabled():
            assert sync_fn("ok") == "ok"
            assert asyncio.run(async_fn("ok")) == "ok"
        assert calls["n"] == 0
        assert sync_fn("checked") == "checked"
        assert calls["n"] == 2
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    # @probe cannot decorate a class; probe.here rejects missing watched_vars when PROBES=1.
    with pytest.raises(TypeError, match="not classes"):
        @probe("Class invariant")
        class BadTarget:
            pass

    with pytest.raises(ValueError, match="watched keyword variable"):
        probe.here("Missing watched")

    # Entry-time UNCHECKED_PROBE with TTY pdb raises ProbeHaltError instead of returning None.
    monkeypatch.delenv("PROBE_NO_PDB", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(pdb.Pdb, "set_trace", lambda self, frame=None: None)
    locked_cfg = _make_config(tmp_path, require_check=True)
    reset_for_testing(config=locked_cfg, transport=httpx.MockTransport(handler))

    @probe("Locked check", id="entry.pdb.locked")
    def never_returns_none() -> str:
        return "real"

    try:
        with pytest.raises(ProbeHaltError):
            never_returns_none()
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    # Store hardening: inner runs/.gitignore, torn trailing line repair, halted not overwritten,
    # and corrupt check_scores skipped gracefully.
    runs_dir = tmp_path / ".probes" / "hardened_runs"
    (tmp_path / ".probes").mkdir(exist_ok=True)
    store = ProbeStore(runs_dir, create=True)
    assert (runs_dir / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"
    store.create_run(
        "01HARDENED0000000000000001",
        started_at="2026-11-18T10:00:00Z",
        cmd="pytest",
        pid=1,
        decision_model="m",
        text_model="t",
    )
    run_file = store.run_path("01HARDENED0000000000000001")
    with run_file.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"event","partial":')
    store.finish_run("01HARDENED0000000000000001", status="halted", halt_call_seq=1)
    store.finish_run("01HARDENED0000000000000001", status="completed")
    with run_file.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "event",
                    "call_seq": 99,
                    "elapsed_ms": 1.0,
                    "started_at": "t",
                    "probe_id": "bad",
                    "site": "here",
                    "probe_type": "decision",
                    "file_line": "a:1",
                    "verdict": "fail",
                    "check_scores": [{"check": "c", "probability": "bad", "passed": False}],
                }
            )
            + "\n"
        )
    run, events = store.get_run_with_events("01HARDENED0000000000000001")
    assert run is not None and run.status == "halted" and run.halt_call_seq == 1
    assert events == []
    assert store.clean_runs(keep=0) == 1


def test_truncation_always_halts_and_strict_api_defaults_true(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    assert TextConfig().max_tokens == 1024
    assert RuntimeConfig().strict_api is True

    monkeypatch.setenv("PROBES", "1")
    monkeypatch.setenv("PROBE_NO_PDB", "1")

    # 1. probe.type.text hitting finish_reason='length' logs PROBE_TRUNCATED and halts even with strict_api=False.
    def trunc_text_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "Check [0] cites Draft 'Hola' which"},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"probabilities": [0.95]})

    cfg_lenient = ProbeConfig(
        models_decision=ModelEndpointConfig(provider="http_decision", url="http://mock.local/decision", model="dec"),
        models_text=ModelEndpointConfig(provider="openai_compat", url="http://mock.local/v1", model="qwen3.5:9b"),
        text=TextConfig(max_tokens=50),
        check=CheckConfig(require_check=False),
        runtime=RuntimeConfig(
            runs_dir=str(tmp_path / ".probes" / "runs_trunc"),
            lock_path=str(tmp_path / ".probes" / "lock.json"),
            strict_api=False,
        ),
        project_root=tmp_path,
    )
    reset_for_testing(config=cfg_lenient, transport=httpx.MockTransport(trunc_text_handler))
    try:
        with pytest.raises(ProbeHaltError, match="increase \\[text\\]\\.max_tokens"):
            probe.here("Grounded draft", id="trunc.text", kind=probe.type.text, draft="Hola")
    finally:
        rt1 = probe.flush_for_testing()
        reset_for_testing()

    assert rt1 is not None
    s1 = ProbeStore(cfg_lenient.resolve_runs_dir(), create=False)
    try:
        ev1 = s1.list_events(rt1.run_id)[0]
    finally:
        s1.close()
    assert ev1.verdict == "error"
    assert ev1.error_code == "PROBE_TRUNCATED"
    assert "Check [0] cites Draft" in (ev1.text_analysis or "")

    # 2. probe.type.decision failure whose explain_failure hits max_tokens preserves partial note instead of None.
    def trunc_explain_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/decision"):
            return httpx.Response(200, json={"probabilities": [0.10]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Output leaked answer 'Quiero"},
                    }
                ]
            },
        )

    cfg_explain = _make_config(tmp_path)
    cfg_explain.text.max_tokens = 123
    reset_for_testing(config=cfg_explain, transport=httpx.MockTransport(trunc_explain_handler))
    try:
        with pytest.raises(ProbeHaltError):
            probe.here("No spoilers", id="trunc.explain", kind=probe.type.decision, reply="Quiero café")
    finally:
        rt2 = probe.flush_for_testing()
        reset_for_testing()

    assert rt2 is not None
    s2 = ProbeStore(cfg_explain.resolve_runs_dir(), create=False)
    try:
        ev2 = s2.list_events(rt2.run_id)[0]
    finally:
        s2.close()
    assert ev2.verdict == "fail"
    assert "Output leaked answer" in (ev2.text_analysis or "")
    assert "truncated at max_tokens=123" in (ev2.text_analysis or "")

    # 3. BACKEND_ERROR halts under default strict_api=True; when strict_api=False prints [PROBE ERROR] to stderr.
    def err500_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream down")

    cfg_strict = _make_config(tmp_path)
    cfg_strict.runtime.runs_dir = str(tmp_path / ".probes" / "runs_strict")
    reset_for_testing(config=cfg_strict, transport=httpx.MockTransport(err500_handler))
    try:
        with pytest.raises(ProbeHaltError, match="HTTP 500"):
            probe.here("Check rule", id="err.strict", x="val")
    finally:
        probe.flush_for_testing()
        reset_for_testing()

    capsys.readouterr()
    cfg_lenient.runtime.runs_dir = str(tmp_path / ".probes" / "runs_err_warn")
    reset_for_testing(config=cfg_lenient, transport=httpx.MockTransport(err500_handler))
    try:
        probe.here("Check rule", id="err.lenient", x="val")
    finally:
        probe.flush_for_testing()
        reset_for_testing()
    captured = capsys.readouterr()
    assert "[PROBE ERROR]" in captured.err
    assert "err.lenient" in captured.err
