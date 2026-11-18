"""Public API for `probe`: @probe, probe.here, probe.type.{decision,text}, ProbeHaltError."""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable, Sequence
from types import FrameType
from typing import Any, ParamSpec, TypeVar

import httpx

from probe.config import ProbeConfig, is_probes_enabled, load_config
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
    if old is not None:
        old.close()
    return None


def _dispatch_probe_event(
    *,
    rt: ProbeRuntime,
    call_seq: int,
    elapsed_ms: float,
    started_at: str,
    probe_id: str,
    site: str,
    probe_type: ProbeType,
    file_line: str,
    checks: tuple[str, ...],
    payload_json: str,
    caller_frame: FrameType | None,
    stack_excerpt: str,
) -> None:
    rt.verify_lock(
        probe_id=probe_id,
        probe_type=probe_type,
        checks=checks,
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        site=site,
        file_line=file_line,
        payload_json=payload_json,
        caller_frame=caller_frame,
    )
    rt.check_rate_limit(
        probe_id=probe_id,
        probe_type=probe_type,
        checks=checks,
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        site=site,
        file_line=file_line,
        payload_json=payload_json,
        caller_frame=caller_frame,
    )
    rt.evaluate_event(
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        probe_id=probe_id,
        site=site,
        probe_type=probe_type,
        file_line=file_line,
        checks=checks,
        payload_json=payload_json,
        caller_frame=caller_frame,
        stack_excerpt=stack_excerpt,
    )


def _wrap_sync_fn(
    fn: Callable[P, R],
    *,
    checks: tuple[str, ...],
    probe_id: str | None,
    probe_type: ProbeType,
) -> Callable[P, R]:
    resolved_id = probe_id or _derive_decorator_id(fn)
    file_line = _decorator_file_line(fn)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_probes_enabled():
            return fn(*args, **kwargs)

        rt = _get_runtime()
        rt.ensure_not_tripped()
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

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        del frame

        stack_excerpt = _short_stack_excerpt()
        exc_caught: BaseException | None = None
        result: Any = None
        has_return = False
        try:
            try:
                result = fn(*args, **kwargs)
                has_return = True
            except ProbeHaltError:
                raise
            except BaseException as exc:
                exc_caught = exc

            payload_json = serialize_call_snapshot(
                inputs_data,
                redactor=rt.redactor,
                return_value=result,
                has_return=has_return,
                exception=exc_caught,
                max_depth=caps.max_depth,
                max_items=caps.max_items,
                max_str_bytes=caps.max_str_bytes,
            )
            _dispatch_probe_event(
                rt=rt,
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
                stack_excerpt=stack_excerpt,
            )
        finally:
            del caller_frame

        if exc_caught is not None:
            raise exc_caught
        return result

    return wrapper


def _wrap_async_fn(
    fn: Callable[P, Any],
    *,
    checks: tuple[str, ...],
    probe_id: str | None,
    probe_type: ProbeType,
) -> Callable[P, Any]:
    resolved_id = probe_id or _derive_decorator_id(fn)
    file_line = _decorator_file_line(fn)

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        if not is_probes_enabled():
            return await fn(*args, **kwargs)

        rt = _get_runtime()
        rt.ensure_not_tripped()
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

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        del frame

        stack_excerpt = _short_stack_excerpt()
        exc_caught: BaseException | None = None
        result: Any = None
        has_return = False
        try:
            try:
                result = await fn(*args, **kwargs)
                has_return = True
            except ProbeHaltError:
                raise
            except BaseException as exc:
                exc_caught = exc

            payload_json = serialize_call_snapshot(
                inputs_data,
                redactor=rt.redactor,
                return_value=result,
                has_return=has_return,
                exception=exc_caught,
                max_depth=caps.max_depth,
                max_items=caps.max_items,
                max_str_bytes=caps.max_str_bytes,
            )
            _dispatch_probe_event(
                rt=rt,
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
                stack_excerpt=stack_excerpt,
            )
        finally:
            del caller_frame

        if exc_caught is not None:
            raise exc_caught
        return result

    return wrapper


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
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        norm_checks = normalize_checks(checks)
        norm_kind = normalize_probe_type(kind)

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            if not is_probes_enabled():
                return fn
            if inspect.iscoroutinefunction(fn):
                return _wrap_async_fn(
                    fn,
                    checks=norm_checks,
                    probe_id=id,
                    probe_type=norm_kind,
                )
            return _wrap_sync_fn(
                fn,
                checks=norm_checks,
                probe_id=id,
                probe_type=norm_kind,
            )

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

        norm_checks = normalize_checks(checks)
        norm_kind = normalize_probe_type(kind)

        rt = _get_runtime()
        rt.ensure_not_tripped()
        call_seq, elapsed_ms, started_at = rt.next_call_meta()

        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame is not None else None
        del frame

        resolved_id = id or _derive_here_id(caller_frame)
        file_line = _here_file_line(caller_frame)
        stack_excerpt = _short_stack_excerpt()

        caps = rt.config.runtime
        payload_json = serialize_here_snapshot(
            watched_vars,
            rt.redactor,
            max_depth=caps.max_depth,
            max_items=caps.max_items,
            max_str_bytes=caps.max_str_bytes,
        )
        try:
            _dispatch_probe_event(
                rt=rt,
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
                stack_excerpt=stack_excerpt,
            )
        finally:
            del caller_frame

    @staticmethod
    def reset_for_testing(
        *,
        config: ProbeConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        reset_for_testing(config=config, transport=transport)

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
type = ProbeType

__all__ = [
    "ProbeHaltError",
    "ProbeType",
    "here",
    "probe",
    "reset_for_testing",
    "type",
]
