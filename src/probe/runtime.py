"""Probe runtime: call_seq tracking, rate guard, async worker, verdict pipeline, halt switch, and atexit drain."""

from __future__ import annotations

import _thread
import atexit
import collections
import concurrent.futures
import hashlib
import inspect
import json
import os
import pdb
import shlex
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any, ParamSpec, TypeVar

import httpx

from probe.backends import BackendError, CheckScore, DecisionBackend, TextBackend
from probe.config import (
    ProbeConfig,
    is_skip_lock,
    load_config,
    new_ulid,
    utc_now_iso,
)
from probe.redact import SecretRedactor
from probe.store import ProbeStore

P = ParamSpec("P")
R = TypeVar("R")


class ProbeType(StrEnum):
    """Evaluation mode for a probe."""

    decision = "decision"
    text = "text"


class ProbeHaltError(RuntimeError):
    """Raised when a probe fails, triggers PROBE_MISUSE, matches sync='when ...', or lacks lock approval."""


def normalize_checks(checks: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a single check string or sequence of check strings into a non-empty tuple."""
    if isinstance(checks, str):
        items = (checks,)
    else:
        items = tuple(str(c) for c in checks)
    if not items or any(not c.strip() for c in items):
        raise ValueError("Probe checks must contain at least one non-empty expectation string")
    return items


def normalize_probe_type(kind: ProbeType | str) -> ProbeType:
    """Validate and coerce kind into ProbeType."""
    if isinstance(kind, ProbeType):
        return kind
    try:
        return ProbeType(str(kind).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported probe kind: {kind!r} (expected probe.type.decision or probe.type.text)") from exc


def compute_checks_hash(kind: ProbeType | str, checks: Sequence[str]) -> str:
    """Compute canonical SHA-256 hex digest for (kind, checks) shared between check and runtime."""
    payload = json.dumps(
        {"kind": str(kind), "checks": list(checks)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_sync_mode(sync: bool | str) -> str:
    if isinstance(sync, bool):
        return "sync" if sync else "async"
    return f"when:{sync}"


def _derive_decorator_id(fn: Callable[..., Any]) -> str:
    mod = getattr(fn, "__module__", "") or "__main__"
    qual = getattr(fn, "__qualname__", getattr(fn, "__name__", "unknown"))
    return f"{mod}.{qual}"


def _derive_here_id(frame: FrameType | None) -> str:
    if frame is None:
        return "unknown:L0"
    mod = str(frame.f_globals.get("__name__", "__main__") or "__main__")
    code = frame.f_code
    qual = getattr(code, "co_qualname", None) or code.co_name
    if qual == "<module>":
        return f"{mod}:L{frame.f_lineno}"
    return f"{mod}.{qual}:L{frame.f_lineno}"


def _decorator_file_line(fn: Callable[..., Any]) -> str:
    try:
        src_file = inspect.getsourcefile(fn) or fn.__code__.co_filename
        _, lineno = inspect.getsourcelines(fn)
        return f"{src_file}:{lineno}"
    except Exception:
        code = getattr(fn, "__code__", None)
        if code is not None:
            return f"{code.co_filename}:{code.co_firstlineno}"
        return "unknown:0"


def _here_file_line(frame: FrameType | None) -> str:
    if frame is None:
        return "unknown:0"
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def _short_stack_excerpt(limit: int = 8) -> str:
    frames = traceback.extract_stack()[:-2]
    if limit > 0:
        frames = frames[-limit:]
    return "".join(traceback.format_list(frames)).strip()


class ProbeRuntime:
    """Manages SQLite run lifecycle, redaction, rate limiting, worker execution, and halting."""

    def __init__(
        self,
        *,
        config: ProbeConfig | None = None,
        transport: httpx.BaseTransport | None = None,
        interrupt_main_fn: Callable[[], None] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.cwd = self.config.project_root
        self.run_id = new_ulid()
        self.started_at = utc_now_iso()
        self._start_mono = time.monotonic()

        self.store = ProbeStore(self.config.resolve_db_path(), create=True)
        cmd = " ".join(shlex.quote(a) for a in sys.argv) if sys.argv else "python"
        self.store.create_run(
            self.run_id,
            started_at=self.started_at,
            cmd=cmd,
            pid=os.getpid(),
            decision_model=f"{self.config.models_decision.provider}:{self.config.models_decision.model}",
            text_model=f"{self.config.models_text.provider}:{self.config.models_text.model}",
        )

        self.redactor = SecretRedactor.from_environment(self.config.redact, cwd=Path.cwd())
        self.decision_backend = DecisionBackend(
            self.config.models_decision,
            self.config.decision,
            transport=transport,
        )
        self.text_backend = TextBackend(
            self.config.models_text,
            self.config.text,
            transport=transport,
        )
        self._interrupt_main = interrupt_main_fn or _thread.interrupt_main

        self._lock_cache: dict[str, dict[str, Any]] | None = None
        self._seq_lock = threading.Lock()
        self._next_seq = 0

        self._rate_lock = threading.Lock()
        self._call_windows: dict[str, collections.deque[float]] = {}
        self._call_totals: dict[str, int] = {}
        self._tripped_probes: set[str] = set()

        self._state_lock = threading.Lock()
        self._tripped = False
        self._halt_reason = ""
        self._draining = False
        self._closed = False
        self._halt_event = threading.Event()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.config.runtime.max_concurrency),
            thread_name_prefix="probe-worker",
        )
        self._futures: set[concurrent.futures.Future[None]] = set()
        self._atexit_registered = False

    def register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def next_call_meta(self) -> tuple[int, float, str]:
        """Claim atomic call_seq, elapsed_ms, and UTC started_at timestamp on call entry."""
        with self._seq_lock:
            self._next_seq += 1
            seq = self._next_seq
        elapsed_ms = round((time.monotonic() - self._start_mono) * 1000.0, 3)
        return seq, elapsed_ms, utc_now_iso()

    def ensure_not_tripped(self) -> None:
        with self._state_lock:
            if self._tripped:
                raise ProbeHaltError(self._halt_reason or "Probe runtime halted")

    def verify_lock(
        self,
        *,
        probe_id: str,
        probe_type: ProbeType,
        checks: tuple[str, ...],
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        site: str,
        file_line: str,
        sync_mode: str,
        payload_json: str,
    ) -> None:
        """Verify probe exists with matching checks_hash in .probes/lock.json unless skipped."""
        if not self.config.check.require_check or is_skip_lock():
            return

        if self._lock_cache is None:
            lock_path = self.config.resolve_lock_path()
            if lock_path.is_file():
                try:
                    raw = json.loads(lock_path.read_text(encoding="utf-8"))
                    probes_map = raw.get("probes") if isinstance(raw, dict) else {}
                    self._lock_cache = probes_map if isinstance(probes_map, dict) else {}
                except Exception:
                    self._lock_cache = {}
            else:
                self._lock_cache = {}

        expected_hash = compute_checks_hash(probe_type, checks)
        entry = self._lock_cache.get(probe_id)
        if isinstance(entry, dict) and entry.get("checks_hash") == expected_hash:
            return

        msg = (
            f"Probe '{probe_id}' failed lock check (.probes/lock.json missing or out of date). "
            "Run `probe check` or set PROBE_SKIP_LOCK=1."
        )
        self._record_and_halt_sync_error(
            probe_id=probe_id,
            probe_type=probe_type,
            checks=checks,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            site=site,
            file_line=file_line,
            sync_mode=sync_mode,
            payload_json=payload_json,
            error_code="UNCHECKED_PROBE",
            analysis=msg,
            stack_excerpt=_short_stack_excerpt(),
        )

    def check_rate_limit(
        self,
        *,
        probe_id: str,
        probe_type: ProbeType,
        checks: tuple[str, ...],
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        site: str,
        file_line: str,
        sync_mode: str,
        payload_json: str,
    ) -> None:
        """Enforce sliding-window and per-run caps; trip PROBE_MISUSE before calling models."""
        rl = self.config.rate_limit
        now = time.monotonic()
        breach_reason: str | None = None

        with self._rate_lock:
            if probe_id in self._tripped_probes:
                breach_reason = f"PROBE_MISUSE: circuit breaker already tripped for '{probe_id}'"
            else:
                window = self._call_windows.setdefault(probe_id, collections.deque())
                cutoff = now - rl.window_seconds
                while window and window[0] < cutoff:
                    window.popleft()
                window.append(now)

                total = self._call_totals.get(probe_id, 0) + 1
                self._call_totals[probe_id] = total

                if total > rl.max_calls_per_run:
                    self._tripped_probes.add(probe_id)
                    breach_reason = (
                        f"PROBE_MISUSE: '{probe_id}' exceeded max_calls_per_run "
                        f"({total} > {rl.max_calls_per_run})"
                    )
                elif len(window) > rl.max_calls:
                    self._tripped_probes.add(probe_id)
                    breach_reason = (
                        f"PROBE_MISUSE: '{probe_id}' exceeded rate limit "
                        f"({len(window)} calls in {rl.window_seconds:.1f}s > {rl.max_calls})"
                    )

        if breach_reason is not None:
            self._record_and_halt_sync_error(
                probe_id=probe_id,
                probe_type=probe_type,
                checks=checks,
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                site=site,
                file_line=file_line,
                sync_mode=sync_mode,
                payload_json=payload_json,
                error_code="PROBE_MISUSE",
                analysis=breach_reason,
                stack_excerpt=_short_stack_excerpt(),
            )

    def _record_and_halt_sync_error(
        self,
        *,
        probe_id: str,
        probe_type: ProbeType,
        checks: tuple[str, ...],
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        site: str,
        file_line: str,
        sync_mode: str,
        payload_json: str,
        error_code: str,
        analysis: str,
        stack_excerpt: str,
    ) -> None:
        self.store.insert_event(
            run_id=self.run_id,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            probe_id=probe_id,
            site=site,
            probe_type=probe_type.value,
            file_line=file_line,
            sync_mode=sync_mode,
            checks=checks,
            payload_json=payload_json,
            check_scores=(),
            verdict="error",
            error_code=error_code,
            text_analysis=analysis,
            stack_excerpt=stack_excerpt,
            latency_ms=0.0,
        )
        self._trigger_halt(
            call_seq=call_seq,
            probe_id=probe_id,
            file_line=file_line,
            summary=analysis,
            is_sync=True,
            caller_frame=None,
        )

    def evaluate_event(
        self,
        *,
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        probe_id: str,
        site: str,
        probe_type: ProbeType,
        file_line: str,
        sync: bool | str,
        checks: tuple[str, ...],
        payload_json: str,
        is_sync_exec: bool,
        caller_frame: FrameType | None = None,
        stack_excerpt: str | None = None,
    ) -> None:
        """Run decision/text grading for one event, persist to SQLite, and halt on failure/trigger."""
        with self._state_lock:
            if self._tripped:
                return

        t0 = time.perf_counter()
        analysis_text: str | None = None
        scores: list[CheckScore] = []
        verdict = "pass"
        error_code: str | None = None
        sync_triggered = False

        try:
            if probe_type == ProbeType.text:
                analysis_text = self.text_backend.analyze_snapshot(
                    probe_id=probe_id,
                    checks=checks,
                    snapshot_json=payload_json,
                )
                scores = self.decision_backend.evaluate_checks(
                    probe_id=probe_id,
                    checks=checks,
                    snapshot_json=payload_json,
                    analysis=analysis_text,
                )
            else:
                scores = self.decision_backend.evaluate_checks(
                    probe_id=probe_id,
                    checks=checks,
                    snapshot_json=payload_json,
                )

            all_passed = all(s.passed for s in scores)
            verdict = "pass" if all_passed else "fail"

            if verdict == "fail" and probe_type == ProbeType.decision and self.config.text.explain_decision_failures:
                try:
                    analysis_text = self.text_backend.explain_failure(
                        probe_id=probe_id,
                        scores=scores,
                        snapshot_json=payload_json,
                    )
                except BackendError:
                    analysis_text = None

            if isinstance(sync, str) and verdict == "pass":
                sync_triggered = self.decision_backend.evaluate_condition(
                    probe_id=probe_id,
                    condition=sync,
                    snapshot_json=payload_json,
                    analysis=analysis_text or "",
                )
        except BackendError as exc:
            verdict = "error"
            error_code = "BACKEND_ERROR"
            analysis_text = str(exc)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        self.store.insert_event(
            run_id=self.run_id,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            probe_id=probe_id,
            site=site,
            probe_type=probe_type.value,
            file_line=file_line,
            sync_mode=_format_sync_mode(sync),
            checks=checks,
            payload_json=payload_json,
            check_scores=scores,
            verdict=verdict,
            error_code=error_code,
            text_analysis=analysis_text,
            stack_excerpt=stack_excerpt if (verdict in {"fail", "error"} or sync_triggered) else None,
            latency_ms=latency_ms,
        )

        should_halt = (
            verdict == "fail"
            or sync_triggered
            or (verdict == "error" and self.config.runtime.strict_api)
        )
        if should_halt:
            if verdict == "fail":
                failed = [s.check for s in scores if not s.passed]
                reason = f"Check(s) failed: {'; '.join(failed)}"
                if analysis_text:
                    reason = f"{reason}\n  analysis: {analysis_text}"
            elif sync_triggered:
                reason = f"Sync breakpoint triggered ({sync})"
            else:
                reason = analysis_text or "Backend error with strict_api=true"

            self._trigger_halt(
                call_seq=call_seq,
                probe_id=probe_id,
                file_line=file_line,
                summary=reason,
                is_sync=is_sync_exec,
                caller_frame=caller_frame,
            )

    def submit_async(self, **kwargs: Any) -> None:
        fut = self._executor.submit(self.evaluate_event, **kwargs)
        with self._state_lock:
            self._futures.add(fut)
        fut.add_done_callback(lambda done: self._discard_future(done))

    def _discard_future(self, fut: concurrent.futures.Future[None]) -> None:
        with self._state_lock:
            self._futures.discard(fut)

    def _trigger_halt(
        self,
        *,
        call_seq: int,
        probe_id: str,
        file_line: str,
        summary: str,
        is_sync: bool,
        caller_frame: FrameType | None,
    ) -> None:
        halt_msg = f"[PROBE HALT] call #{call_seq:03d} {probe_id} ({file_line}): {summary}"
        with self._state_lock:
            if self._tripped:
                if is_sync:
                    raise ProbeHaltError(self._halt_reason or halt_msg)
                return
            self._tripped = True
            self._halt_reason = halt_msg
            pending = list(self._futures)
            draining = self._draining

        for fut in pending:
            fut.cancel()

        self.store.finish_run(
            self.run_id,
            status="halted",
            finished_at=utc_now_iso(),
            halt_call_seq=call_seq,
        )
        print(halt_msg, file=sys.stderr, flush=True)
        self._halt_event.set()

        if is_sync:
            use_pdb = (
                caller_frame is not None
                and os.environ.get("PROBE_NO_PDB", "0").strip() != "1"
                and bool(getattr(sys.stdin, "isatty", lambda: False)())
                and bool(getattr(sys.stdout, "isatty", lambda: False)())
            )
            if use_pdb:
                pdb.Pdb().set_trace(caller_frame)
                return
            raise ProbeHaltError(halt_msg)

        if not draining:
            try:
                self._interrupt_main()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Drain in-flight async evaluations up to drain_timeout_s on process exit."""
        with self._state_lock:
            if self._closed:
                return
            self._draining = True
            pending = list(self._futures)

        deadline = time.monotonic() + max(0.0, self.config.runtime.drain_timeout_s)
        while pending and not self._halt_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, not_done = concurrent.futures.wait(
                pending,
                timeout=min(0.05, remaining),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            _ = done
            pending = list(not_done)

        with self._state_lock:
            tripped = self._tripped
            self._closed = True

        self._executor.shutdown(wait=False, cancel_futures=tripped)
        if not tripped:
            self.store.finish_run(
                self.run_id,
                status="completed",
                finished_at=utc_now_iso(),
            )
        self.decision_backend.close()
        self.text_backend.close()
        self.store.close()
