"""Runtime secret collection (.env* + os.environ), request-field masking, and bounded serialization."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from probe.config import RedactConfig

REDACTED = "[REDACTED]"

DEFAULT_MASK_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "passphrase",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
    }
)

_SECRET_ENV_KEY_TOKENS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASS",
    "PWD",
    "AUTH",
    "DSN",
    "DATABASE_URL",
    "REDIS_URL",
    "AMQP_URL",
    "PRIVATE",
    "CREDENTIAL",
    "WEBHOOK",
    "COOKIE",
    "SESSION",
)

_NON_SECRET_ENV_SUFFIXES: tuple[str, ...] = (
    "_ENV",
    "_HOST",
    "_PORT",
    "_LEVEL",
    "_MODEL",
    "_DIR",
    "_PATH",
    "_REGION",
)

_NON_SECRET_EXACT_KEYS: frozenset[str] = frozenset(
    {
        "ENV",
        "ENVIRONMENT",
        "LOG_LEVEL",
        "HOST",
        "PORT",
        "MODEL",
        "DEBUG",
        "TZ",
        "LANG",
        "LC_ALL",
        "HOME",
        "USER",
        "PWD",
        "SHELL",
        "TERM",
    }
)

_NON_SECRET_VALUES: frozenset[str] = frozenset(
    {
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        "none",
        "null",
        "production",
        "development",
        "staging",
        "testing",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "standard",
        "detailed",
        "concise",
    }
)

_EXAMPLE_DOTENV_SUFFIXES: tuple[str, ...] = (
    ".example",
    ".sample",
    ".template",
    ".dist",
)

_SECRET_QUERY_KEYS: frozenset[str] = frozenset(
    {"apikey", "api_key", "token", "secret", "password", "key", "auth"}
)

_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_PATH_PREFIX_RE = re.compile(r"^(?:/|\./|\.\./|~/|[A-Za-z]:\\)")
_EXPORT_PREFIX_RE = re.compile(r"^export\s+")


def _is_secret_env_key(name: str) -> bool:
    upper = name.strip().upper()
    if not upper or upper in _NON_SECRET_EXACT_KEYS:
        return False
    if upper.endswith(_NON_SECRET_ENV_SUFFIXES) and not any(
        tok in upper for tok in ("SECRET", "TOKEN", "PASS", "PRIVATE", "AUTH")
    ):
        return False
    return any(tok in upper for tok in _SECRET_ENV_KEY_TOKENS)


def _is_non_secret_dotenv_key(name: str) -> bool:
    upper = name.strip().upper()
    if upper in _NON_SECRET_EXACT_KEYS:
        return True
    return upper.endswith(_NON_SECRET_ENV_SUFFIXES) and not any(
        tok in upper for tok in ("SECRET", "TOKEN", "PASS", "PRIVATE", "AUTH", "KEY")
    )


def _is_url_like(val: str) -> bool:
    return "://" in val and not val.startswith("://")


def _extract_url_secrets(val: str) -> tuple[bool, list[str]]:
    """Return (url_itself_has_secret, extracted_secret_parts)."""
    parsed = urlparse(val)
    if not parsed.scheme or not parsed.netloc:
        return False, []
    found: list[str] = []
    has_secret = False
    if parsed.password and len(parsed.password) >= 4:
        has_secret = True
        found.append(parsed.password)
    if parsed.query:
        qs = parse_qs(parsed.query, keep_blank_values=False)
        for qk, qvals in qs.items():
            if qk.strip().lower() in _SECRET_QUERY_KEYS:
                for qv in qvals:
                    if len(qv) >= 4:
                        has_secret = True
                        found.append(qv)
    if has_secret:
        found.append(val)
    return has_secret, found


def _should_keep_secret_value(val: str, *, min_len: int = 8) -> bool:
    stripped = val.strip()
    if len(stripped) < min_len:
        return False
    lower = stripped.lower()
    if lower in _NON_SECRET_VALUES:
        return False
    if _NUMBER_RE.match(stripped):
        return False
    if _PATH_PREFIX_RE.match(stripped) and "://" not in stripped:
        return False
    if _is_url_like(stripped):
        has_secret, _ = _extract_url_secrets(stripped)
        return has_secret
    return True


def _strip_inline_comment(raw: str) -> str:
    for idx, ch in enumerate(raw):
        if ch == "#" and idx > 0 and raw[idx - 1].isspace():
            return raw[:idx].rstrip()
    return raw.rstrip()


def _parse_dotenv_file(path: Path) -> list[tuple[str, str]]:
    """Parse KEY=VALUE pairs from a dotenv file cleanly without executing shell code."""
    entries: list[tuple[str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return entries

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = _EXPORT_PREFIX_RE.sub("", stripped, count=1)
        if "=" not in stripped:
            continue
        key, raw_val = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = raw_val.strip()
        if val and val[0] in {"'", '"'}:
            quote = val[0]
            if quote == '"':
                chars: list[str] = []
                escape = False
                for ch in val[1:]:
                    if escape:
                        if ch == "n":
                            chars.append("\n")
                        elif ch == "t":
                            chars.append("\t")
                        else:
                            chars.append(ch)
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == quote:
                        break
                    else:
                        chars.append(ch)
                val = "".join(chars)
            else:
                end_idx = val.find("'", 1)
                val = val[1:end_idx] if end_idx != -1 else val[1:]
        else:
            val = _strip_inline_comment(val)
        entries.append((key, val))
    return entries


def _is_example_dotenv(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suf) for suf in _EXAMPLE_DOTENV_SUFFIXES)


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


class SecretRedactor:
    """Collects runtime secrets once and masks secret values and credential fields."""

    def __init__(self, secrets: Sequence[str] = (), extra_keys: Sequence[str] = ()) -> None:
        unique_secrets = sorted(
            {s for s in secrets if s},
            key=len,
            reverse=True,
        )
        self.secrets: tuple[str, ...] = tuple(unique_secrets)
        self.mask_keys: frozenset[str] = DEFAULT_MASK_KEYS | frozenset(
            _normalize_key(k) for k in extra_keys if str(k).strip()
        )
        if self.secrets:
            pattern = "|".join(re.escape(s) for s in self.secrets)
            self._pattern: re.Pattern[str] | None = re.compile(pattern)
        else:
            self._pattern = None

    @classmethod
    def from_environment(
        cls,
        config: RedactConfig | None = None,
        cwd: Path | None = None,
    ) -> SecretRedactor:
        """Collect secret strings from .env* files, [redact].env_files, and secret-named os.environ vars."""
        cfg = config or RedactConfig()
        work_dir = (cwd or Path.cwd()).resolve()
        collected: set[str] = set()

        # 1. Discover dotenv files: cwd/.env* plus configured env_files.
        candidate_files: list[Path] = []
        seen_paths: set[Path] = set()

        if work_dir.is_dir():
            for entry in sorted(work_dir.glob(".env*")):
                if entry.is_file() and not _is_example_dotenv(entry):
                    resolved = entry.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        candidate_files.append(entry)

        for raw_file in cfg.env_files:
            p = Path(raw_file).expanduser()
            if not p.is_absolute():
                p = work_dir / p
            if p.is_file() and not _is_example_dotenv(p):
                resolved = p.resolve()
                if resolved not in seen_paths:
                    seen_paths.add(resolved)
                    candidate_files.append(p)

        for env_file in candidate_files:
            for key, val in _parse_dotenv_file(env_file):
                if _is_url_like(val):
                    has_secret, url_secrets = _extract_url_secrets(val)
                    if has_secret:
                        collected.update(url_secrets)
                    continue
                if _is_non_secret_dotenv_key(key):
                    continue
                if _should_keep_secret_value(val, min_len=8):
                    collected.add(val.strip())

        # 2. Scan os.environ for secret-named keys and URLs containing passwords/secret params.
        for key, val in os.environ.items():
            if not val:
                continue
            if _is_url_like(val):
                has_secret, url_secrets = _extract_url_secrets(val)
                if has_secret:
                    collected.update(url_secrets)
                continue
            if _is_secret_env_key(key) and _should_keep_secret_value(val, min_len=8):
                collected.add(val.strip())

        return cls(secrets=sorted(collected), extra_keys=cfg.extra_keys)

    def scrub_text(self, text: str) -> str:
        """Replace collected runtime secret substrings with '[REDACTED]'."""
        if not text or self._pattern is None:
            return text
        return self._pattern.sub(REDACTED, text)

    def should_mask_key(self, key: str) -> bool:
        """Return True when a field/parameter name should always be replaced with '[REDACTED]'."""
        return _normalize_key(key) in self.mask_keys


def _truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    suffix = "...[truncated]"
    keep = max(0, max_bytes - len(suffix))
    clipped = encoded[:keep].decode("utf-8", errors="ignore")
    return f"{clipped}{suffix}"


def _serialize_bounded(
    value: Any,
    redactor: SecretRedactor,
    *,
    max_depth: int,
    max_items: int,
    max_str_bytes: int,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        scrubbed = redactor.scrub_text(value)
        return _truncate_utf8(scrubbed, max_str_bytes)
    if isinstance(value, (bytes, bytearray)):
        decoded = bytes(value).decode("utf-8", errors="replace")
        scrubbed = redactor.scrub_text(decoded)
        return _truncate_utf8(scrubbed, max_str_bytes)

    if depth >= max_depth:
        preview = redactor.scrub_text(repr(value))
        return _truncate_utf8(preview, min(max_str_bytes, 256))

    obj_id = id(value)
    if obj_id in seen:
        return "<circular>"

    seen.add(obj_id)
    try:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= max_items:
                    out["__truncated_items__"] = max(0, len(value) - max_items)
                    break
                key_str = redactor.scrub_text(str(k))
                if redactor.should_mask_key(key_str):
                    out[key_str] = REDACTED
                else:
                    out[key_str] = _serialize_bounded(
                        v,
                        redactor,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_str_bytes=max_str_bytes,
                        depth=depth + 1,
                        seen=seen,
                    )
            return out

        if isinstance(value, (list, tuple, set, frozenset)):
            items: list[Any] = []
            seq = list(value)
            for idx, item in enumerate(seq):
                if idx >= max_items:
                    items.append(f"...[{len(seq) - max_items} more items]")
                    break
                items.append(
                    _serialize_bounded(
                        item,
                        redactor,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_str_bytes=max_str_bytes,
                        depth=depth + 1,
                        seen=seen,
                    )
                )
            return items

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _serialize_bounded(
                dataclasses.asdict(value),
                redactor,
                max_depth=max_depth,
                max_items=max_items,
                max_str_bytes=max_str_bytes,
                depth=depth,
                seen=seen,
            )

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return _serialize_bounded(
                    model_dump(),
                    redactor,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_str_bytes=max_str_bytes,
                    depth=depth,
                    seen=seen,
                )
            except Exception:
                pass

        if hasattr(value, "__dict__") and not isinstance(value, type):
            attrs = {k: v for k, v in vars(value).items() if not k.startswith("_")}
            if attrs:
                return _serialize_bounded(
                    attrs,
                    redactor,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_str_bytes=max_str_bytes,
                    depth=depth,
                    seen=seen,
                )

        preview = redactor.scrub_text(repr(value))
        return _truncate_utf8(preview, max_str_bytes)
    finally:
        seen.discard(obj_id)


def serialize_call_inputs(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    redactor: SecretRedactor,
    *,
    max_depth: int = 4,
    max_items: int = 25,
    max_str_bytes: int = 4096,
) -> dict[str, Any]:
    """Bind function arguments by parameter name, mask request secrets, and bound sizes."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        raw_inputs: dict[str, Any] = dict(bound.arguments)
    except (TypeError, ValueError):
        raw_inputs = {}
        if args:
            raw_inputs["args"] = list(args)
        if kwargs:
            raw_inputs["kwargs"] = dict(kwargs)

    return _serialize_bounded(
        raw_inputs,
        redactor,
        max_depth=max_depth,
        max_items=max_items,
        max_str_bytes=max_str_bytes,
    )


def serialize_call_snapshot(
    inputs_data: dict[str, Any],
    *,
    redactor: SecretRedactor,
    return_value: Any = None,
    has_return: bool = False,
    exception: BaseException | None = None,
    max_depth: int = 4,
    max_items: int = 25,
    max_str_bytes: int = 4096,
) -> str:
    """Build redacted JSON payload combining entry inputs and exit return/exception."""
    payload: dict[str, Any] = {"inputs": inputs_data}
    if exception is not None:
        exc_msg = redactor.scrub_text(f"{type(exception).__name__}: {exception}")
        payload["exception"] = _truncate_utf8(exc_msg, max_str_bytes)
    elif has_return:
        payload["output"] = _serialize_bounded(
            return_value,
            redactor,
            max_depth=max_depth,
            max_items=max_items,
            max_str_bytes=max_str_bytes,
        )
    return json.dumps(payload, ensure_ascii=False)


def serialize_here_snapshot(
    watched_vars: dict[str, Any],
    redactor: SecretRedactor,
    *,
    max_depth: int = 4,
    max_items: int = 25,
    max_str_bytes: int = 4096,
) -> str:
    """Serialize watched variables for probe.here(...) with redaction and size bounds."""
    payload = _serialize_bounded(
        {"watched": watched_vars},
        redactor,
        max_depth=max_depth,
        max_items=max_items,
        max_str_bytes=max_str_bytes,
    )
    return json.dumps(payload, ensure_ascii=False)
