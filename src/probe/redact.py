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
from urllib.parse import parse_qs, unquote, urlparse

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
        "api_key",
        "apikey",
        "x_api_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
        "secret_key",
        "auth_token",
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
    "CRED",
    "BEARER",
    "SIGNING",
    "ENCRYPT",
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


def _is_url_like(val: str) -> bool:
    return "://" in val and not val.startswith("://")


def _extract_url_secrets(val: str) -> tuple[bool, list[str]]:
    """Return (url_itself_has_secret, extracted_secret_parts)."""
    try:
        parsed = urlparse(val)
    except ValueError:
        return False, []
    if not parsed.scheme or not parsed.netloc:
        return False, []
    found: list[str] = []
    has_secret = False
    if parsed.password and len(parsed.password) >= 4:
        has_secret = True
        found.append(parsed.password)
        unquoted_pw = unquote(parsed.password)
        if unquoted_pw and unquoted_pw != parsed.password:
            found.append(unquoted_pw)
    elif parsed.username and len(parsed.username) >= 8:
        has_secret = True
        found.append(parsed.username)
        unquoted_user = unquote(parsed.username)
        if unquoted_user and unquoted_user != parsed.username:
            found.append(unquoted_user)
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


def _should_keep_secret_value(val: str, *, min_len: int = 8, allow_plain_url: bool = False) -> bool:
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
        if allow_plain_url:
            return True
        has_secret, _ = _extract_url_secrets(stripped)
        return has_secret
    return True


def _strip_inline_comment(raw: str) -> str:
    for idx, ch in enumerate(raw):
        if ch == "#" and idx > 0 and raw[idx - 1].isspace():
            return raw[:idx].rstrip()
    return raw.rstrip()


def _parse_dotenv_file(path: Path) -> list[tuple[str, str]]:
    """Parse KEY=VALUE pairs (including multiline quoted values) cleanly without shell execution."""
    entries: list[tuple[str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return entries

    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        if text[idx] == "#":
            nl = text.find("\n", idx)
            idx = n if nl == -1 else nl + 1
            continue

        nl = text.find("\n", idx)
        line_end = n if nl == -1 else nl
        eq = text.find("=", idx)
        if eq == -1 or eq > line_end:
            idx = line_end + 1
            continue

        raw_key = text[idx:eq].strip()
        key = _EXPORT_PREFIX_RE.sub("", raw_key, count=1).strip()
        if not key:
            idx = line_end + 1
            continue

        p = eq + 1
        while p < n and text[p] in " \t":
            p += 1

        if p < n and text[p] in {"'", '"'}:
            quote = text[p]
            p += 1
            if quote == '"':
                chars: list[str] = []
                escape = False
                while p < n:
                    ch = text[p]
                    p += 1
                    if escape:
                        if ch == "n":
                            chars.append("\n")
                        elif ch == "t":
                            chars.append("\t")
                        elif ch == "r":
                            chars.append("\r")
                        else:
                            chars.append(ch)
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        break
                    else:
                        chars.append(ch)
                val = "".join(chars)
                rem_nl = text.find("\n", p)
                idx = n if rem_nl == -1 else rem_nl + 1
            else:
                end_idx = text.find("'", p)
                if end_idx != -1:
                    val = text[p:end_idx]
                    rem_nl = text.find("\n", end_idx + 1)
                    idx = n if rem_nl == -1 else rem_nl + 1
                else:
                    val = text[p:line_end].rstrip("\r")
                    idx = line_end + 1
        else:
            raw_val = text[p:line_end].rstrip("\r")
            val = _strip_inline_comment(raw_val)
            idx = line_end + 1

        entries.append((key, val))
    return entries


def _is_example_dotenv(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suf) for suf in _EXAMPLE_DOTENV_SUFFIXES)


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _collect_env_pair(key: str, val: str, collected: set[str]) -> None:
    stripped = val.strip()
    if not stripped:
        return
    if _is_url_like(stripped):
        has_secret, url_secrets = _extract_url_secrets(stripped)
        if has_secret:
            collected.update(url_secrets)
        if _is_secret_env_key(key) and _should_keep_secret_value(stripped, min_len=8, allow_plain_url=True):
            collected.add(stripped)
        return
    if _is_secret_env_key(key) and _should_keep_secret_value(stripped, min_len=8):
        collected.add(stripped)


class SecretRedactor:
    """Collects runtime secrets once and masks secret values and credential fields."""

    def __init__(self, secrets: Sequence[str] = (), extra_keys: Sequence[str] = ()) -> None:
        expanded: set[str] = set()
        for raw_s in secrets:
            if not raw_s:
                continue
            expanded.add(raw_s)
            escaped = json.dumps(raw_s, ensure_ascii=False)[1:-1]
            if escaped and escaped != raw_s:
                expanded.add(escaped)
        unique_secrets = sorted(expanded, key=len, reverse=True)
        self.secrets: tuple[str, ...] = tuple(unique_secrets)
        self.max_secret_len: int = len(self.secrets[0]) if self.secrets else 0
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
        extra_dirs: Sequence[Path] = (),
    ) -> SecretRedactor:
        """Collect secret strings from .env* files, [redact].env_files, and secret-named os.environ vars."""
        cfg = config or RedactConfig()
        search_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        for d in [cwd or Path.cwd(), *extra_dirs]:
            resolved_dir = Path(d).resolve()
            if resolved_dir not in seen_dirs:
                seen_dirs.add(resolved_dir)
                search_dirs.append(resolved_dir)

        collected: set[str] = set()
        candidate_files: list[Path] = []
        seen_paths: set[Path] = set()

        for base_dir in search_dirs:
            if base_dir.is_dir():
                for entry in sorted(base_dir.glob(".env*")):
                    if entry.is_file() and not _is_example_dotenv(entry):
                        resolved = entry.resolve()
                        if resolved not in seen_paths:
                            seen_paths.add(resolved)
                            candidate_files.append(entry)

            for raw_file in cfg.env_files:
                p = Path(raw_file).expanduser()
                if not p.is_absolute():
                    p = base_dir / p
                if p.is_file() and not _is_example_dotenv(p):
                    resolved = p.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        candidate_files.append(p)

        for env_file in candidate_files:
            for key, val in _parse_dotenv_file(env_file):
                _collect_env_pair(key, val, collected)

        for key, val in os.environ.items():
            if val:
                _collect_env_pair(key, val, collected)

        return cls(secrets=sorted(collected), extra_keys=cfg.extra_keys)

    def scrub_text(self, text: str) -> str:
        """Replace collected runtime secret substrings with '[REDACTED]'."""
        if not text or self._pattern is None:
            return text
        return self._pattern.sub(REDACTED, text)

    def should_mask_key(self, key: str) -> bool:
        """Return True when a field/parameter name should always be replaced with '[REDACTED]'."""
        return _normalize_key(key) in self.mask_keys


def _sanitize_utf8(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _truncate_utf8(text: str, max_bytes: int, *, force_marker: bool = False) -> str:
    clean = _sanitize_utf8(text)
    encoded = clean.encode("utf-8")
    if len(encoded) <= max_bytes and not force_marker:
        return clean
    suffix = "...[truncated]"
    keep = max(0, max_bytes - len(suffix))
    clipped = encoded[:keep].decode("utf-8", errors="ignore")
    return f"{clipped}{suffix}"


def _scrub_and_truncate(text: str, redactor: SecretRedactor, max_bytes: int) -> str:
    clean = _sanitize_utf8(text)
    scan_limit = max_bytes + redactor.max_secret_len + 64
    pre_clipped = len(clean) > scan_limit
    if pre_clipped:
        clean = clean[:scan_limit]
    scrubbed = redactor.scrub_text(clean)
    return _truncate_utf8(scrubbed, max_bytes, force_marker=pre_clipped and len(scrubbed.encode("utf-8")) > max_bytes)


_MISSING = object()


def _extract_object_attrs(value: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if hasattr(value, "__dict__"):
        for k, v in vars(value).items():
            if not str(k).startswith("_"):
                attrs[str(k)] = v
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__")
        if not slots:
            continue
        slot_names = (slots,) if isinstance(slots, str) else slots
        for slot in slot_names:
            name = str(slot)
            if name.startswith("_") or name in {"__dict__", "__weakref__"} or name in attrs:
                continue
            attr_val = getattr(value, name, _MISSING)
            if attr_val is not _MISSING:
                attrs[name] = attr_val
    return attrs


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
        return _scrub_and_truncate(value, redactor, max_str_bytes)
    if isinstance(value, (bytes, bytearray)):
        raw_cap = max_str_bytes + redactor.max_secret_len + 64
        decoded = bytes(value[:raw_cap]).decode("utf-8", errors="replace")
        return _scrub_and_truncate(decoded, redactor, max_str_bytes)

    if depth >= max_depth:
        if isinstance(value, Mapping):
            return f"<dict len={len(value)} max_depth>"
        if isinstance(value, (list, tuple, set, frozenset)):
            return f"<{type(value).__name__} len={len(value)} max_depth>"
        return f"<{type(value).__qualname__} max_depth>"

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
                key_str = _sanitize_utf8(redactor.scrub_text(str(k)))
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

        fields_tuple = getattr(value, "_fields", None)
        if isinstance(value, tuple) and isinstance(fields_tuple, tuple):
            named_map = {str(f): getattr(value, str(f), None) for f in fields_tuple}
            return _serialize_bounded(
                named_map,
                redactor,
                max_depth=max_depth,
                max_items=max_items,
                max_str_bytes=max_str_bytes,
                depth=depth,
                seen=seen,
            )

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
            dc_fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
            return _serialize_bounded(
                dc_fields,
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

        if not isinstance(value, type):
            attrs = _extract_object_attrs(value)
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

        try:
            raw_repr = repr(value)
        except Exception:
            raw_repr = f"<unrepresentable {type(value).__qualname__}>"
        return _scrub_and_truncate(raw_repr, redactor, max_str_bytes)
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
        exc_msg = f"{type(exception).__name__}: {exception}"
        payload["exception"] = _scrub_and_truncate(exc_msg, redactor, max_str_bytes)
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
    serialized_watched = _serialize_bounded(
        watched_vars,
        redactor,
        max_depth=max_depth,
        max_items=max_items,
        max_str_bytes=max_str_bytes,
        depth=0,
    )
    return json.dumps({"watched": serialized_watched}, ensure_ascii=False)
