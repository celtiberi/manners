"""Probe runtime: synchronous inline grading, call_seq tracking, rate guard, and halt switch."""

from __future__ import annotations

import atexit
import collections
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
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any, ParamSpec, TypeVar

import httpx

from probe.backends import BackendError, CheckScore, DecisionBackend, TextBackend
from probe.config import (
    ProbeConfig,
    is_no_pdb,
    is_skip_lock,
    load_config,
    new_ulid,
    utc_now_iso,
)
from probe.redact import SecretRedactor, _scrub_and_truncate, serialize_call_snapshot
from probe.store import ProbeStore

P = ParamSpec("P")
R = TypeVar("R")


class ProbeType(StrEnum):
    """Evaluation mode for a probe."""

    decision = "decision"
    text = "text"


class ProbeHaltError(RuntimeError):
    """Raised when a probe fails, triggers PROBE_MISUSE, or lacks lock approval."""


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


def resolve_module_name(file_path: Path, root: Path | None = None) -> str:
    """Resolve dotted Python module name matching importable package & PEP 420 namespace layouts."""
    resolved_file = file_path.resolve()
    resolved_root = root.resolve() if root is not None else None

    pkg_parts: list[str] = []
    cur = resolved_file.parent
    reached_root = False
    while True:
        if resolved_root is not None and cur == resolved_root:
            if (cur / "__init__.py").is_file():
                pkg_parts.append(cur.name)
            reached_root = True
            break
        if not (cur / "__init__.py").is_file():
            break
        pkg_parts.append(cur.name)
        if cur.parent == cur:
            break
        cur = cur.parent
    pkg_parts.reverse()

    if resolved_root is not None and not reached_root:
        src_dir = (resolved_root / "src").resolve()
        ns_base: Path | None = None
        if (resolved_root / "src").is_dir() and resolved_file.is_relative_to(src_dir):
            ns_base = src_dir
        elif resolved_file.is_relative_to(resolved_root):
            ns_base = resolved_root
        if ns_base is not None:
            rel = resolved_file.relative_to(ns_base)
            dir_parts = rel.parent.parts
            if dir_parts and all(part.isidentifier() for part in dir_parts):
                pkg_parts = list(dir_parts)

    if resolved_file.stem == "__init__":
        return ".".join(pkg_parts) if pkg_parts else resolved_file.stem
    parts = [*pkg_parts, resolved_file.stem]
    return ".".join(parts)


def _resolve_runtime_module(
    mod_name: str,
    src_file: str | None,
    project_root: Path | None,
) -> str:
    if mod_name == "__main__":
        main_mod = sys.modules.get("__main__")
        spec = getattr(main_mod, "__spec__", None)
        spec_name = getattr(spec, "name", None) if spec is not None else None
        if isinstance(spec_name, str) and spec_name and spec_name != "__main__":
            return spec_name
        if src_file:
            p = Path(src_file)
            if p.is_file():
                return resolve_module_name(p, project_root)
    return mod_name or "__main__"


def _normalize_here_qualname(raw_qual: str) -> str:
    parts = [p for p in str(raw_qual).split(".") if p and p not in {"<listcomp>", "<dictcomp>", "<setcomp>"}]
    cleaned: list[str] = []
    for part in parts:
        if part == "<locals>" and cleaned and cleaned[-1] == "<locals>":
            continue
        cleaned.append(part)
    while cleaned and cleaned[-1] == "<locals>":
        cleaned.pop()
    if not cleaned:
        return "<module>"
    return ".".join(cleaned)


def _rel_file_line(src_file: str, lineno: int, project_root: Path | None = None) -> str:
    if project_root is not None and src_file and src_file != "unknown":
        try:
            rel = Path(src_file).resolve().relative_to(project_root.resolve())
            return f"{rel.as_posix()}:{lineno}"
        except (OSError, ValueError):
            pass
    return f"{src_file}:{lineno}"


def _derive_decorator_id(fn: Callable[..., Any], *, project_root: Path | None = None) -> str:
    target = inspect.unwrap(fn)
    raw_mod = getattr(target, "__module__", "") or getattr(type(target), "__module__", "") or "__main__"
    qual = getattr(target, "__qualname__", getattr(target, "__name__", "")) or type(target).__qualname__
    src_file: str | None = None
    try:
        src_file = inspect.getsourcefile(target)
    except Exception:
        code = getattr(target, "__code__", None)
        src_file = code.co_filename if code is not None else None
    mod = _resolve_runtime_module(str(raw_mod), src_file, project_root)
    return f"{mod}.{qual}"


def _derive_here_id(frame: FrameType | None, *, project_root: Path | None = None) -> str:
    if frame is None:
        return "unknown:L0"
    raw_mod = str(frame.f_globals.get("__name__", "__main__") or "__main__")
    code = frame.f_code
    mod = _resolve_runtime_module(raw_mod, code.co_filename, project_root)
    raw_qual = getattr(code, "co_qualname", None) or code.co_name
    qual = _normalize_here_qualname(str(raw_qual))
    if qual == "<module>":
        return f"{mod}:L{frame.f_lineno}"
    return f"{mod}.{qual}:L{frame.f_lineno}"


def _decorator_file_line(fn: Callable[..., Any], *, project_root: Path | None = None) -> str:
    target = inspect.unwrap(fn)
    code = getattr(target, "__code__", None)
    try:
        src_file = inspect.getsourcefile(target) or (code.co_filename if code is not None else None)
    except Exception:
        src_file = code.co_filename if code is not None else None
    if code is not None and src_file:
        return _rel_file_line(src_file, code.co_firstlineno, project_root)
    try:
        resolved_file = inspect.getsourcefile(target)
        _, lineno = inspect.getsourcelines(target)
        if resolved_file:
            return _rel_file_line(resolved_file, lineno, project_root)
    except Exception:
        pass
    return "unknown:0"


def _here_file_line(frame: FrameType | None, *, project_root: Path | None = None) -> str:
    if frame is None:
        return "unknown:0"
    return _rel_file_line(frame.f_code.co_filename, frame.f_lineno, project_root)


@dataclass(slots=True, frozen=True)
class GradedEvent:
    """In-memory model grading result ready to be persisted and checked for halts."""

    probe_id: str
    site: str
    probe_type: ProbeType
    file_line: str
    checks: tuple[str, ...]
    payload_json: str
    scores: tuple[CheckScore, ...]
    verdict: str
    error_code: str | None
    analysis_text: str | None
    latency_ms: float


def _short_stack_excerpt(caller_frame: FrameType | None = None, limit: int = 8) -> str:
    frames = traceback.extract_stack(caller_frame) if caller_frame is not None else traceback.extract_stack()[:-2]
    if limit > 0:
        frames = frames[-limit:]
    return "".join(traceback.format_list(frames)).strip()


def _is_asyncio_internal_frame(frame: FrameType) -> bool:
    norm = frame.f_code.co_filename.replace("\\", "/")
    return "/asyncio/" in norm


class ProbeRuntime:
    """Manages JSONL run lifecycle, redaction, rate limiting, inline grading, and halting."""

    def __init__(
        self,
        *,
        config: ProbeConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or load_config()
        self.cwd = self.config.project_root
        self.run_id = new_ulid()
        self.started_at = utc_now_iso()
        self._start_mono = time.monotonic()

        self.store = ProbeStore(self.config.resolve_runs_dir(), create=True)
        try:
            cmd = " ".join(shlex.quote(a) for a in sys.argv) if sys.argv else "python"
            self.store.create_run(
                self.run_id,
                started_at=self.started_at,
                cmd=cmd,
                pid=os.getpid(),
                decision_model=f"{self.config.models_decision.provider}:{self.config.models_decision.model}",
                text_model=f"{self.config.models_text.provider}:{self.config.models_text.model}",
            )

            self.redactor = SecretRedactor.from_environment(
                self.config.redact,
                cwd=self.config.project_root,
                extra_dirs=(Path.cwd(),),
            )
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
        except BaseException:
            self.store.close()
            raise

        self._lock_cache: dict[str, dict[str, Any]] | None = None
        self._lock_cache_lock = threading.Lock()
        self._seq_lock = threading.Lock()
        self._next_seq = 0

        self._rate_lock = threading.Lock()
        self._call_windows: dict[str, collections.deque[float]] = {}
        self._call_totals: dict[str, int] = {}
        self._tripped_probes: set[str] = set()

        self._state_lock = threading.Lock()
        self._tripped = False
        self._halt_reason = ""
        self._closed = False
        self._atexit_registered = False

    @property
    def is_tripped(self) -> bool:
        with self._state_lock:
            return self._tripped

    def register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self.close)
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
        payload_json: str,
        caller_frame: FrameType | None = None,
        watched_keys: tuple[str, ...] = (),
        param_names: tuple[str, ...] | None = None,
    ) -> bool:
        """Verify probe matches lockfile (checks_hash, site, params/watched_keys, file) unless skipped."""
        if not self.config.check.require_check or is_skip_lock(self.cwd):
            return False

        with self._lock_cache_lock:
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
            entry = self._lock_cache.get(probe_id)

        expected_hash = compute_checks_hash(probe_type, checks)
        entry_site = entry.get("site") if isinstance(entry, dict) else None
        site_ok = entry_site == site
        watched_ok = True
        if site == "here" and isinstance(entry, dict):
            locked_keys = entry.get("watched_keys")
            watched_ok = isinstance(locked_keys, list) and {str(k) for k in locked_keys} == set(watched_keys)

        params_ok = True
        if site == "decorator" and param_names is not None and isinstance(entry, dict):
            locked_params = entry.get("param_names")
            if isinstance(locked_params, list):
                params_ok = [str(k) for k in locked_params] == list(param_names)

        file_ok = True
        if isinstance(entry, dict) and isinstance(entry.get("file_line"), str):
            entry_fl = entry["file_line"]
            if ":" in entry_fl and ":" in file_line:
                entry_file = entry_fl.rsplit(":", 1)[0]
                runtime_file = file_line.rsplit(":", 1)[0]
                if not Path(runtime_file).is_absolute() and not Path(entry_file).is_absolute():
                    file_ok = entry_file == runtime_file

        if (
            isinstance(entry, dict)
            and entry.get("checks_hash") == expected_hash
            and site_ok
            and watched_ok
            and params_ok
            and file_ok
        ):
            return False

        msg = (
            f"Probe '{probe_id}' failed lock check (.probes/lock.json missing or out of date). "
            "Run `probe check` or set PROBE_SKIP_LOCK=1."
        )
        return self._record_and_halt_error(
            probe_id=probe_id,
            probe_type=probe_type,
            checks=checks,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            site=site,
            file_line=file_line,
            payload_json=payload_json,
            error_code="UNCHECKED_PROBE",
            analysis=msg,
            stack_excerpt=_short_stack_excerpt(caller_frame),
            caller_frame=caller_frame,
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
        payload_json: str,
        caller_frame: FrameType | None = None,
    ) -> bool:
        """Enforce sliding-window and optional per-run caps; trip PROBE_MISUSE before calling models."""
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

                if rl.max_calls_per_run >= 0 and total > rl.max_calls_per_run:
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
            return self._record_and_halt_error(
                probe_id=probe_id,
                probe_type=probe_type,
                checks=checks,
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                site=site,
                file_line=file_line,
                payload_json=payload_json,
                error_code="PROBE_MISUSE",
                analysis=breach_reason,
                stack_excerpt=_short_stack_excerpt(caller_frame),
                caller_frame=caller_frame,
            )
        return False

    def _record_and_halt_error(
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
        payload_json: str,
        error_code: str,
        analysis: str,
        stack_excerpt: str,
        caller_frame: FrameType | None,
    ) -> bool:
        self.store.insert_event(
            run_id=self.run_id,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            probe_id=probe_id,
            site=site,
            probe_type=probe_type.value,
            file_line=file_line,
            checks=checks,
            payload_json=payload_json,
            check_scores=(),
            verdict="error",
            error_code=error_code,
            text_analysis=analysis,
            stack_excerpt=stack_excerpt,
            latency_ms=0.0,
        )
        return self._trigger_halt(
            call_seq=call_seq,
            probe_id=probe_id,
            file_line=file_line,
            summary=analysis,
            caller_frame=caller_frame,
        )

    def record_target_exception(
        self,
        *,
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        probe_id: str,
        site: str,
        probe_type: ProbeType,
        file_line: str,
        checks: tuple[str, ...],
        inputs_data: dict[str, Any],
        exception: Exception,
        exc_traceback: str,
    ) -> None:
        """Log an unhandled target-function exception without grading or masking the crash."""
        if self.is_tripped:
            return
        caps = self.config.runtime
        payload_json = serialize_call_snapshot(
            inputs_data,
            redactor=self.redactor,
            exception=exception,
            max_depth=caps.max_depth,
            max_items=caps.max_items,
            max_str_bytes=caps.max_str_bytes,
        )
        summary = _scrub_and_truncate(f"{type(exception).__name__}: {exception}", self.redactor, caps.max_str_bytes)
        clean_tb = _scrub_and_truncate(exc_traceback.strip(), self.redactor, max(caps.max_str_bytes, 16384))
        self.store.insert_event(
            run_id=self.run_id,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            probe_id=probe_id,
            site=site,
            probe_type=probe_type.value,
            file_line=file_line,
            checks=checks,
            payload_json=payload_json,
            check_scores=(),
            verdict="error",
            error_code="TARGET_EXCEPTION",
            text_analysis=summary,
            stack_excerpt=clean_tb or None,
            latency_ms=0.0,
        )

    def grade_event(
        self,
        *,
        probe_id: str,
        site: str,
        probe_type: ProbeType,
        file_line: str,
        checks: tuple[str, ...],
        payload_json: str,
    ) -> GradedEvent:
        """Execute decision/text model evaluation without mutating run logs or halting state."""
        t0 = time.perf_counter()
        analysis_text: str | None = None
        scores: list[CheckScore] = []
        verdict = "pass"
        error_code: str | None = None

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
        except BackendError as exc:
            verdict = "error"
            error_code = "BACKEND_ERROR"
            analysis_text = _scrub_and_truncate(str(exc), self.redactor, self.config.runtime.max_str_bytes)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return GradedEvent(
            probe_id=probe_id,
            site=site,
            probe_type=probe_type,
            file_line=file_line,
            checks=checks,
            payload_json=payload_json,
            scores=tuple(scores),
            verdict=verdict,
            error_code=error_code,
            analysis_text=analysis_text,
            latency_ms=latency_ms,
        )

    def record_and_maybe_halt(
        self,
        graded: GradedEvent,
        *,
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        caller_frame: FrameType | None = None,
        stack_excerpt: str | None = None,
    ) -> None:
        """Persist a GradedEvent in the JSONL store and halt on fail or strict_api error."""
        self.ensure_not_tripped()
        final_stack = stack_excerpt
        if graded.verdict in {"fail", "error"} and not final_stack:
            final_stack = _short_stack_excerpt(caller_frame)

        self.store.insert_event(
            run_id=self.run_id,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            probe_id=graded.probe_id,
            site=graded.site,
            probe_type=graded.probe_type.value,
            file_line=graded.file_line,
            checks=graded.checks,
            payload_json=graded.payload_json,
            check_scores=graded.scores,
            verdict=graded.verdict,
            error_code=graded.error_code,
            text_analysis=graded.analysis_text,
            stack_excerpt=final_stack if graded.verdict in {"fail", "error"} else None,
            latency_ms=graded.latency_ms,
        )

        should_halt = (
            graded.verdict == "fail"
            or (graded.verdict == "error" and self.config.runtime.strict_api)
        )
        if should_halt:
            if graded.verdict == "fail":
                failed = [s.check for s in graded.scores if not s.passed]
                reason = f"Check(s) failed: {'; '.join(failed)}"
                if graded.analysis_text:
                    reason = f"{reason}\n  analysis: {graded.analysis_text}"
            else:
                reason = graded.analysis_text or "Backend error with strict_api=true"

            self._trigger_halt(
                call_seq=call_seq,
                probe_id=graded.probe_id,
                file_line=graded.file_line,
                summary=reason,
                caller_frame=caller_frame,
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
        checks: tuple[str, ...],
        payload_json: str,
        caller_frame: FrameType | None = None,
        stack_excerpt: str | None = None,
    ) -> None:
        """Run decision/text grading inline for one event, persist to JSONL, and halt on fail."""
        self.ensure_not_tripped()
        graded = self.grade_event(
            probe_id=probe_id,
            site=site,
            probe_type=probe_type,
            file_line=file_line,
            checks=checks,
            payload_json=payload_json,
        )
        self.record_and_maybe_halt(
            graded,
            call_seq=call_seq,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            caller_frame=caller_frame,
            stack_excerpt=stack_excerpt,
        )

    def _trigger_halt(
        self,
        *,
        call_seq: int,
        probe_id: str,
        file_line: str,
        summary: str,
        caller_frame: FrameType | None,
    ) -> bool:
        halt_msg = f"[PROBE HALT] call #{call_seq:03d} {probe_id} ({file_line}): {summary}"
        with self._state_lock:
            is_closed = self._closed
            first_halt = not self._tripped
            self._tripped = True
            if first_halt:
                self._halt_reason = halt_msg

        if first_halt and not is_closed:
            self.store.finish_run(
                self.run_id,
                status="halted",
                finished_at=utc_now_iso(),
                halt_call_seq=call_seq,
            )
        print(halt_msg, file=sys.stderr, flush=True)

        use_pdb = (
            caller_frame is not None
            and threading.current_thread() is threading.main_thread()
            and "PYTEST_CURRENT_TEST" not in os.environ
            and not is_no_pdb(self.cwd)
            and bool(getattr(sys.stdin, "isatty", lambda: False)())
            and bool(getattr(sys.stdout, "isatty", lambda: False)())
            and not _is_asyncio_internal_frame(caller_frame)
        )
        if use_pdb:
            pdb.Pdb().set_trace(caller_frame)
            return True
        raise ProbeHaltError(halt_msg)

    def close(self) -> None:
        """Finalize run status and close model clients and store on exit."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            tripped = self._tripped
            atexit_was_registered = self._atexit_registered
            self._atexit_registered = False

        if atexit_was_registered:
            atexit.unregister(self.close)

        try:
            if not tripped:
                self.store.finish_run(
                    self.run_id,
                    status="completed",
                    finished_at=utc_now_iso(),
                )
        finally:
            try:
                self.decision_backend.close()
            finally:
                try:
                    self.text_backend.close()
                finally:
                    self.store.close()
