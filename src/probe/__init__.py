"""Public API for `probe`: @probe, probe.here, probe.type.{decision,text}, ProbeHaltError."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import functools
import inspect
import json
import os
import threading
import traceback
from collections.abc import Callable, Iterator, Sequence
from types import FrameType
from typing import Any, ParamSpec, TypeVar

import httpx

from probe.config import (
    _OVERRIDE_CTX,
    ProbeConfig,
    clear_env_cache,
    is_probes_enabled,
    load_config,
    pop_process_override,
    push_process_override,
    reset_config_locks_after_fork,
)
from probe.redact import serialize_call_inputs, serialize_call_snapshot, serialize_here_snapshot
from probe.runtime import (
    ProbeHaltError,
    ProbeRuntime,
    ProbeType,
    _decorator_file_line,
    _derive_decorator_id,
    _derive_here_id,
    _here_file_line,
    _short_stack_excerpt,
    normalize_checks,
    normalize_probe_type,
)

F = TypeVar("F")
P = ParamSpec("P")
R = TypeVar("R")

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: ProbeRuntime | None = None
_TEST_CONFIG: ProbeConfig | None = None
_TEST_TRANSPORT: httpx.BaseTransport | None = None


def _get_runtime() -> ProbeRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            cfg = _TEST_CONFIG or load_config()
            rt = ProbeRuntime(
                config=cfg,
                transport=_TEST_TRANSPORT,
            )
            rt.register_atexit()
            _RUNTIME = rt
        return _RUNTIME


def reset_for_testing(
    *,
    config: ProbeConfig | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ProbeRuntime | None:
    """Close any active ProbeRuntime and optionally inject test config/transport."""
    global _RUNTIME, _TEST_CONFIG, _TEST_TRANSPORT
    with _RUNTIME_LOCK:
        old = _RUNTIME
        _RUNTIME = None
        _TEST_CONFIG = config
        _TEST_TRANSPORT = transport
    clear_env_cache()
    if old is not None:
        old.close()
    return None


def _reset_child_runtime() -> None:
    global _RUNTIME, _RUNTIME_LOCK
    _RUNTIME = None
    _RUNTIME_LOCK = threading.Lock()
    reset_config_locks_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_child_runtime)


def _callable_param_names(fn: Callable[..., Any]) -> tuple[str, ...] | None:
    try:
        return tuple(inspect.signature(inspect.unwrap(fn)).parameters.keys())
    except (TypeError, ValueError):
        return None


def _prepare_decorator_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    checks: tuple[str, ...],
    probe_id: str | None,
    probe_type: ProbeType,
    caller_frame: FrameType | None,
) -> tuple[ProbeRuntime, str, str, int, float, str, dict[str, Any], bool]:
    """Validate lock & rate limits and snapshot inputs BEFORE calling wrapped `fn`."""
    rt = _get_runtime()
    rt.ensure_not_tripped()
    resolved_id = probe_id or _derive_decorator_id(fn, project_root=rt.cwd)
    file_line = _decorator_file_line(fn, project_root=rt.cwd)
    call_seq, elapsed_ms, started_at = rt.next_call_meta()
    caps = rt.config.runtime
    inputs_data = serialize_call_inputs(
        fn,
        args,
        kwargs,
        rt.redactor,
        max_depth=caps.max_depth,
        max_items=caps.max_items,
        max_str_bytes=caps.max_str_bytes,
    )
    entry_payload_json = json.dumps({"inputs": inputs_data}, ensure_ascii=False)

    entered_pdb_lock = rt.verify_lock(
        probe_id=resolved_id,
        probe_type=probe_type,
        checks=checks,
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        site="decorator",
        file_line=file_line,
        payload_json=entry_payload_json,
        caller_frame=caller_frame,
        param_names=_callable_param_names(fn),
    )
    if entered_pdb_lock or rt.is_tripped:
        raise ProbeHaltError(rt._halt_reason or "Probe halted before target call")
    entered_pdb_rate, skip_grading = rt.check_rate_limit(
        probe_id=resolved_id,
        probe_type=probe_type,
        checks=checks,
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        site="decorator",
        file_line=file_line,
        payload_json=entry_payload_json,
        caller_frame=caller_frame,
    )
    if entered_pdb_rate or rt.is_tripped:
        raise ProbeHaltError(rt._halt_reason or "Probe halted before target call")
    return rt, resolved_id, file_line, call_seq, elapsed_ms, started_at, inputs_data, skip_grading


def _wrap_sync_fn(
    fn: Callable[P, R],
    *,
    checks: tuple[str, ...],
    probe_id: str | None,
    probe_type: ProbeType,
) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_probes_enabled():
            return fn(*args, **kwargs)

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        del frame

        try:
            rt, resolved_id, file_line, call_seq, elapsed_ms, started_at, inputs_data, skip_grading = (
                _prepare_decorator_call(
                    fn,
                    args,
                    kwargs,
                    checks=checks,
                    probe_id=probe_id,
                    probe_type=probe_type,
                    caller_frame=caller_frame,
                )
            )
            if skip_grading:
                return fn(*args, **kwargs)

            try:
                result = fn(*args, **kwargs)
            except ProbeHaltError:
                raise
            except Exception as exc:
                rt.record_target_exception(
                    call_seq=call_seq,
                    elapsed_ms=elapsed_ms,
                    started_at=started_at,
                    probe_id=resolved_id,
                    site="decorator",
                    probe_type=probe_type,
                    file_line=file_line,
                    checks=checks,
                    inputs_data=inputs_data,
                    exception=exc,
                    exc_traceback=traceback.format_exc(),
                )
                raise

            caps = rt.config.runtime
            payload_json = serialize_call_snapshot(
                inputs_data,
                redactor=rt.redactor,
                return_value=result,
                has_return=True,
                max_depth=caps.max_depth,
                max_items=caps.max_items,
                max_str_bytes=caps.max_str_bytes,
            )
            rt.evaluate_event(
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                probe_id=resolved_id,
                site="decorator",
                probe_type=probe_type,
                file_line=file_line,
                checks=checks,
                payload_json=payload_json,
                caller_frame=caller_frame,
            )
            return result
        finally:
            del caller_frame

    return wrapper


def _wrap_async_fn(
    fn: Callable[P, Any],
    *,
    checks: tuple[str, ...],
    probe_id: str | None,
    probe_type: ProbeType,
) -> Callable[P, Any]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        if not is_probes_enabled():
            return await fn(*args, **kwargs)

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        del frame

        try:
            rt, resolved_id, file_line, call_seq, elapsed_ms, started_at, inputs_data, skip_grading = (
                _prepare_decorator_call(
                    fn,
                    args,
                    kwargs,
                    checks=checks,
                    probe_id=probe_id,
                    probe_type=probe_type,
                    caller_frame=caller_frame,
                )
            )
        finally:
            del caller_frame

        if skip_grading:
            return await fn(*args, **kwargs)

        try:
            result = await fn(*args, **kwargs)
        except ProbeHaltError:
            raise
        except Exception as exc:
            rt.record_target_exception(
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                probe_id=resolved_id,
                site="decorator",
                probe_type=probe_type,
                file_line=file_line,
                checks=checks,
                inputs_data=inputs_data,
                exception=exc,
                exc_traceback=traceback.format_exc(),
            )
            raise

        rt.ensure_not_tripped()
        caps = rt.config.runtime
        payload_json = serialize_call_snapshot(
            inputs_data,
            redactor=rt.redactor,
            return_value=result,
            has_return=True,
            max_depth=caps.max_depth,
            max_items=caps.max_items,
            max_str_bytes=caps.max_str_bytes,
        )
        if not rt.should_halt_on_fail():
            after_frame = inspect.currentframe()
            after_caller = after_frame.f_back if after_frame is not None else None
            del after_frame
            try:
                stack_excerpt = _short_stack_excerpt(after_caller)
            finally:
                del after_caller
            rt.evaluate_event(
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                probe_id=resolved_id,
                site="decorator",
                probe_type=probe_type,
                file_line=file_line,
                checks=checks,
                payload_json=payload_json,
                caller_frame=None,
                stack_excerpt=stack_excerpt,
            )
            return result

        if not rt.begin_active_eval():
            return result
        try:
            graded = await asyncio.to_thread(
                rt.grade_event,
                probe_id=resolved_id,
                site="decorator",
                probe_type=probe_type,
                file_line=file_line,
                checks=checks,
                payload_json=payload_json,
            )
            after_frame = inspect.currentframe()
            after_caller = after_frame.f_back if after_frame is not None else None
            del after_frame
            try:
                rt.record_and_maybe_halt(
                    graded,
                    call_seq=call_seq,
                    elapsed_ms=elapsed_ms,
                    started_at=started_at,
                    caller_frame=after_caller,
                )
            finally:
                del after_caller
        finally:
            rt.end_active_eval()
        return result

    return wrapper


def handle_thread_excepthook(
    args: threading.ExceptHookArgs,
    prev_hook: Callable[[threading.ExceptHookArgs], Any] | None,
    *,
    suppress_forward: bool = False,
) -> None:
    """Capture uncaught ProbeHaltError from background threads so flush() surfaces it on main."""
    if isinstance(args.exc_value, ProbeHaltError):
        with _RUNTIME_LOCK:
            rt = _RUNTIME
        if rt is not None:
            rt.record_unhandled_thread_halt(str(args.exc_value))
        if suppress_forward:
            return
    if prev_hook is not None:
        prev_hook(args)


_PREV_THREAD_EXCEPTHOOK = threading.excepthook


def _probe_thread_excepthook(args: threading.ExceptHookArgs) -> None:
    handle_thread_excepthook(args, _PREV_THREAD_EXCEPTHOOK, suppress_forward=False)


threading.excepthook = _probe_thread_excepthook


@contextlib.contextmanager
def _override_scope(enabled_state: bool, scope: str, api_name: str) -> Iterator[None]:
    norm_scope = scope.strip().lower()
    if norm_scope == "process":
        token = push_process_override(enabled_state)
        try:
            yield
        finally:
            pop_process_override(token)
    elif norm_scope == "context":
        token = _OVERRIDE_CTX.set(enabled_state)
        try:
            yield
        finally:
            _OVERRIDE_CTX.reset(token)
    else:
        raise ValueError(f"Invalid {api_name} scope {scope!r}: expected 'process' or 'context'")


class _ProbeAPI:
    """Callable `probe` facade providing `@probe(...)`, `probe.here(...)`, and `probe.type`."""

    type = ProbeType
    ProbeHaltError = ProbeHaltError

    def __call__(
        self,
        checks: str | Sequence[str],
        *,
        id: str | None = None,
        kind: ProbeType | str = ProbeType.decision,
    ) -> Callable[[F], F]:
        norm_checks = normalize_checks(checks)
        norm_kind = normalize_probe_type(kind)

        def decorator(fn: F) -> F:
            desc_type: builtins.type[classmethod[Any, Any, Any]] | builtins.type[staticmethod[Any, Any]] | None = None
            target: Any = fn
            if isinstance(fn, classmethod):
                desc_type = classmethod
                target = fn.__func__
            elif isinstance(fn, staticmethod):
                desc_type = staticmethod
                target = fn.__func__

            if isinstance(target, builtins.type):
                raise TypeError("@probe can only decorate functions or methods, not classes")

            unwrapped = inspect.unwrap(target)
            if inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(unwrapped):
                raise TypeError(
                    "@probe cannot wrap generator functions; use probe.here(...) inside the generator instead"
                )

            is_async = inspect.iscoroutinefunction(unwrapped) or (
                not inspect.isfunction(target)
                and not inspect.ismethod(target)
                and callable(target)
                and inspect.iscoroutinefunction(builtins.type(target).__call__)
            )
            if is_async:
                wrapped: Any = _wrap_async_fn(
                    target,
                    checks=norm_checks,
                    probe_id=id,
                    probe_type=norm_kind,
                )
            else:
                wrapped = _wrap_sync_fn(
                    target,
                    checks=norm_checks,
                    probe_id=id,
                    probe_type=norm_kind,
                )

            if desc_type is not None:
                return desc_type(wrapped)  # type: ignore[return-value]
            return wrapped

        return decorator

    def here(
        self,
        checks: str | Sequence[str],
        *,
        id: str | None = None,
        kind: ProbeType | str = ProbeType.decision,
        **watched_vars: Any,
    ) -> None:
        """Mid-function watchpoint capturing watched_vars and evaluating checks inline."""
        if not is_probes_enabled():
            return

        if not watched_vars:
            raise ValueError("probe.here requires at least one watched keyword variable")

        norm_checks = normalize_checks(checks)
        norm_kind = normalize_probe_type(kind)

        rt = _get_runtime()
        rt.ensure_not_tripped()
        call_seq, elapsed_ms, started_at = rt.next_call_meta()

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        del frame

        resolved_id = id or _derive_here_id(caller_frame, project_root=rt.cwd)
        file_line = _here_file_line(caller_frame, project_root=rt.cwd)

        caps = rt.config.runtime
        payload_json = serialize_here_snapshot(
            watched_vars,
            rt.redactor,
            max_depth=caps.max_depth,
            max_items=caps.max_items,
            max_str_bytes=caps.max_str_bytes,
        )
        try:
            entered_pdb_lock = rt.verify_lock(
                probe_id=resolved_id,
                probe_type=norm_kind,
                checks=norm_checks,
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                site="here",
                file_line=file_line,
                payload_json=payload_json,
                caller_frame=caller_frame,
                watched_keys=tuple(watched_vars.keys()),
            )
            if entered_pdb_lock:
                return
            rt.ensure_not_tripped()
            entered_pdb_rate, skip_grading = rt.check_rate_limit(
                probe_id=resolved_id,
                probe_type=norm_kind,
                checks=norm_checks,
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                site="here",
                file_line=file_line,
                payload_json=payload_json,
                caller_frame=caller_frame,
            )
            if entered_pdb_rate or skip_grading:
                return
            rt.ensure_not_tripped()
            rt.evaluate_event(
                call_seq=call_seq,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
                probe_id=resolved_id,
                site="here",
                probe_type=norm_kind,
                file_line=file_line,
                checks=norm_checks,
                payload_json=payload_json,
                caller_frame=caller_frame,
            )
        finally:
            del caller_frame

    @staticmethod
    def disabled(scope: str = "process") -> contextlib.AbstractContextManager[None]:
        """Temporarily suppress @probe and probe.here ('process' covers threads; 'context' is ContextVar-local)."""
        return _override_scope(False, scope, "disabled")

    @staticmethod
    def enabled(scope: str = "process") -> contextlib.AbstractContextManager[None]:
        """Temporarily enable @probe and probe.here ('process' covers threads; 'context' is ContextVar-local)."""
        return _override_scope(True, scope, "enabled")

    @staticmethod
    def reset_for_testing(
        *,
        config: ProbeConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        reset_for_testing(config=config, transport=transport)

    @staticmethod
    def flush(*, check: bool = True) -> None:
        """Wait for pending soak/background evaluations and surface uncaught thread halts if check=True."""
        with _RUNTIME_LOCK:
            rt = _RUNTIME
        if rt is None:
            return
        drain_timeout = max(rt.config.models_decision.timeout_s, rt.config.models_text.timeout_s) + 5.0
        rt.wait_pending(timeout=drain_timeout)
        halted_t = rt.halted_thread
        if (
            halted_t is not None
            and halted_t is not threading.current_thread()
            and halted_t.is_alive()
            and "ThreadPoolExecutor" not in halted_t.name
        ):
            halted_t.join(timeout=1.0)
        if check:
            unhandled = rt.consume_unhandled_thread_halt()
            if unhandled is not None:
                raise ProbeHaltError(unhandled)

    @staticmethod
    def flush_for_testing() -> ProbeRuntime | None:
        global _RUNTIME
        with _RUNTIME_LOCK:
            rt = _RUNTIME
            _RUNTIME = None
        if rt is not None:
            rt.close()
        return rt


probe = _ProbeAPI()
here = probe.here
disabled = probe.disabled
enabled = probe.enabled
flush = probe.flush
type = ProbeType

__all__ = [
    "ProbeHaltError",
    "ProbeType",
    "disabled",
    "enabled",
    "flush",
    "here",
    "probe",
    "reset_for_testing",
]
