"""Configuration loading, environment helpers, and ID/timestamp utilities."""

from __future__ import annotations

import math
import os
import re
import threading
import time
import tomllib
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_LOCK = threading.Lock()
_LAST_MS = 0
_LAST_RAND = 0

_DEFAULT_USER_CONFIG_PATH = Path.home() / ".config" / "probe" / "config.toml"
_EXPORT_PREFIX_RE = re.compile(r"^export\s+")
_OVERRIDE_CTX: ContextVar[bool | None] = ContextVar("probe_override", default=None)
_DISABLED_LOCK = threading.Lock()
_PROCESS_OVERRIDES: list[tuple[object, bool]] = []
_TRUSTED_DEFAULT_HOSTS = frozenset(
    {
        "api.typesafe.ai",
        "api.deepseek.com",
        "api.openai.com",
        "generativelanguage.googleapis.com",
    }
)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_OPTIONAL_INT_FIELDS = frozenset({"entailment_index", "contradiction_index"})
_ALLOWED_TEXT_DETAILS = frozenset({"concise", "standard", "detailed"})
_ALLOWED_ONNX_OUTPUTS = frozenset({"logits", "probabilities"})


def _strip_inline_comment(raw: str) -> str:
    for idx, ch in enumerate(raw):
        if ch == "#" and idx > 0 and raw[idx - 1].isspace():
            return raw[:idx].rstrip()
    return raw.rstrip()


def parse_dotenv_file(path: Path) -> list[tuple[str, str]]:
    """Parse KEY=VALUE pairs (including multiline quoted values) safely without shell execution."""
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


def _lookup_dotenv_key(env_path: Path, key_name: str) -> str | None:
    """Return stripped value for key_name from a dotenv file, or None if absent."""
    matched: str | None = None
    for k, val in parse_dotenv_file(env_path):
        if k == key_name:
            matched = val.strip()
    return matched


def _parse_dotenv_key(env_path: Path, key_name: str) -> str:
    """Return value for key_name from a dotenv file, or empty string if absent."""
    return _lookup_dotenv_key(env_path, key_name) or ""


def _encode_base32(value: int, length: int) -> str:
    chars = ["0"] * length
    for idx in range(length - 1, -1, -1):
        chars[idx] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(chars)


def new_ulid() -> str:
    """Generate a thread-safe monotonic 26-char Crockford Base32 ULID."""
    global _LAST_MS, _LAST_RAND
    with _ULID_LOCK:
        now_ms = int(time.time() * 1000)
        if now_ms <= _LAST_MS:
            now_ms = _LAST_MS
            _LAST_RAND = (_LAST_RAND + 1) & ((1 << 80) - 1)
            if _LAST_RAND == 0:
                now_ms += 1
                _LAST_MS = now_ms
        else:
            _LAST_MS = now_ms
            _LAST_RAND = int.from_bytes(os.urandom(10), "big")
        return _encode_base32(now_ms, 10) + _encode_base32(_LAST_RAND, 16)


def utc_now_iso() -> str:
    """Return current UTC time formatted as ISO-8601 with microsecond precision and Z suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_ENV_FLAG_CACHE: dict[tuple[str, str], tuple[tuple[int, int, int] | None, str | None]] = {}
_ENV_FLAG_CACHE_LOCK = threading.Lock()
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def push_process_override(enabled_state: bool) -> object:
    """Push a process-wide probe enable/disable override (thread-safe across executors)."""
    token = object()
    with _DISABLED_LOCK:
        _PROCESS_OVERRIDES.append((token, enabled_state))
    return token


def pop_process_override(token: object) -> None:
    """Remove a previously pushed process-wide override by token."""
    with _DISABLED_LOCK:
        for idx in range(len(_PROCESS_OVERRIDES) - 1, -1, -1):
            if _PROCESS_OVERRIDES[idx][0] is token:
                _PROCESS_OVERRIDES.pop(idx)
                break


def reset_config_locks_after_fork() -> None:
    """Reinitialize module locks in a forked child process to prevent deadlocks."""
    global _ULID_LOCK, _ENV_FLAG_CACHE_LOCK, _DISABLED_LOCK
    _ULID_LOCK = threading.Lock()
    _ENV_FLAG_CACHE_LOCK = threading.Lock()
    _DISABLED_LOCK = threading.Lock()
    _PROCESS_OVERRIDES.clear()


def clear_env_cache() -> None:
    """Clear cached .env flag lookups (used after environment or directory changes in tests)."""
    with _ENV_FLAG_CACHE_LOCK:
        _ENV_FLAG_CACHE.clear()


def _read_dotenv_cached(env_path: Path, var_name: str) -> str | None:
    try:
        st = env_path.stat()
        sig: tuple[int, int, int] | None = (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        sig = None

    cache_key = (str(env_path), var_name)
    with _ENV_FLAG_CACHE_LOCK:
        cached = _ENV_FLAG_CACHE.get(cache_key)
        if cached is not None and cached[0] == sig:
            return cached[1]

    parsed = _lookup_dotenv_key(env_path, var_name) if sig is not None else None
    with _ENV_FLAG_CACHE_LOCK:
        _ENV_FLAG_CACHE[cache_key] = (sig, parsed)
    return parsed


def _read_env_flag(var_name: str, cwd: Path | None = None) -> str:
    if var_name in os.environ:
        return os.environ[var_name].strip()

    base_dir = Path(str(cwd)) if cwd is not None else None
    root = find_project_root(base_dir)
    for search_dir in find_env_search_dirs(root):
        parsed = _read_dotenv_cached(search_dir / ".env", var_name)
        if parsed is not None:
            return parsed
    return ""


def is_probes_enabled(cwd: Path | None = None) -> bool:
    """Return True when PROBES is enabled in os.environ or project .env (unless overridden)."""
    ctx_val = _OVERRIDE_CTX.get()
    if ctx_val is not None:
        return ctx_val
    with _DISABLED_LOCK:
        if _PROCESS_OVERRIDES:
            return _PROCESS_OVERRIDES[-1][1]
    return _read_env_flag("PROBES", cwd).lower() in _TRUE_VALUES


def is_skip_lock(cwd: Path | None = None) -> bool:
    """Return True when PROBE_SKIP_LOCK=1 bypasses .probes/lock.json checks."""
    return _read_env_flag("PROBE_SKIP_LOCK", cwd).lower() in _TRUE_VALUES


def is_no_pdb(cwd: Path | None = None) -> bool:
    """Return True when PROBE_NO_PDB=1 disables interactive pdb halts."""
    return _read_env_flag("PROBE_NO_PDB", cwd).lower() in _TRUE_VALUES


def is_no_halt(cwd: Path | None = None) -> bool:
    """Return True when PROBE_NO_HALT=1 keeps execution running after fail/error events."""
    return _read_env_flag("PROBE_NO_HALT", cwd).lower() in _TRUE_VALUES


@dataclass(slots=True)
class ModelEndpointConfig:
    """Endpoint and model settings for either the decision or text model role."""

    provider: str
    url: str = ""
    model: str = ""
    api_key_env: str = ""
    api_key: str = ""
    timeout_s: float = 30.0
    onnx_path: str = ""
    tokenizer_path: str = ""
    entailment_index: int | None = None
    contradiction_index: int | None = None
    onnx_output: str = "logits"
    max_length: int = 512
    reasoning_effort: str = ""
    env_search_dirs: tuple[Path, ...] = ()
    env_files: tuple[str, ...] = (".env",)

    def resolve_api_key(self) -> str | None:
        """Return configured literal key, os.environ value, or dotenv value within project_root."""
        if self.api_key:
            return self.api_key
        if not self.api_key_env:
            return None
        if self.api_key_env in os.environ:
            env_val = os.environ[self.api_key_env].strip()
            return env_val or None

        seen: set[Path] = set()
        for base_dir in self.env_search_dirs:
            resolved_base = base_dir.resolve()
            for raw_name in self.env_files:
                candidate = Path(raw_name).expanduser()
                if not candidate.is_absolute():
                    candidate = resolved_base / candidate
                resolved = candidate.resolve()
                if not resolved.is_relative_to(resolved_base):
                    continue
                if resolved in seen or not candidate.is_file():
                    continue
                seen.add(resolved)
                val = _parse_dotenv_key(candidate, self.api_key_env)
                if val:
                    return val
        return None


@dataclass(slots=True)
class DecisionConfig:
    """Probability thresholding for decision checks."""

    pass_threshold: float = 0.50


@dataclass(slots=True)
class TextConfig:
    """Settings controlling text-model analysis, failure explanations, and pre-run audits."""

    detail: str = "standard"  # "concise" | "standard" | "detailed"
    max_tokens: int = 1024
    explain_decision_failures: bool = True
    system_prompt: str = ""


@dataclass(slots=True)
class RateLimitConfig:
    """Per-probe sliding-window and optional per-run call caps (-1 = unlimited)."""

    max_calls: int = 20
    window_seconds: float = 10.0
    max_calls_per_run: int = -1


@dataclass(slots=True)
class CheckConfig:
    """Pre-run probe check budgets and lockfile enforcement."""

    max_checks: int = 12
    max_check_chars: int = 200
    require_check: bool = True
    max_workers: int = 4


@dataclass(slots=True)
class RedactConfig:
    """Secret redaction sources and extra field names."""

    env_files: tuple[str, ...] = (".env",)
    extra_keys: tuple[str, ...] = ()


@dataclass(slots=True)
class RuntimeConfig:
    """Storage paths, snapshot caps, and strict API / halt handling."""

    runs_dir: str = ".probes/runs"
    lock_path: str = ".probes/lock.json"
    strict_api: bool = True
    halt_on_fail: bool = True
    max_depth: int = 4
    max_items: int = 25
    max_str_bytes: int = 4096
    max_workers: int = 4


@dataclass(slots=True)
class ProbeConfig:
    """Top-level resolved configuration for probe."""

    models_decision: ModelEndpointConfig = field(
        default_factory=lambda: ModelEndpointConfig(
            provider="typesafe_jev",
            url="https://api.typesafe.ai/v1/systemone",
            model="jev-latest",
            api_key_env="TYPESAFE_API_KEY",
        )
    )
    models_text: ModelEndpointConfig = field(
        default_factory=lambda: ModelEndpointConfig(
            provider="openai_compat",
            url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        )
    )
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    check: CheckConfig = field(default_factory=CheckConfig)
    redact: RedactConfig = field(default_factory=RedactConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    domain_context: str = ""
    project_root: Path = field(default_factory=Path.cwd)

    def resolve_runs_dir(self) -> Path:
        path = Path(self.runtime.runs_dir)
        return path if path.is_absolute() else (self.project_root / path)

    def resolve_lock_path(self) -> Path:
        path = Path(self.runtime.lock_path)
        return path if path.is_absolute() else (self.project_root / path)


def _find_auto_project_root(start: Path | None = None) -> Path:
    """Locate nearest directory with pyproject.toml, .probes, or .git; fallback to start."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".probes").is_dir():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return cur


def find_env_search_dirs(start: Path) -> tuple[Path, ...]:
    """Return ordered directories (nearest-first) from start up to the enclosing .git root."""
    cur = start.resolve()
    home = Path.home().resolve()
    chain: list[Path] = []
    for candidate in (cur, *cur.parents):
        chain.append(candidate)
        if (candidate / ".git").exists():
            if candidate == home and cur != home:
                return (_find_auto_project_root(cur),)
            return tuple(chain)
        if candidate == home:
            break

    auto_root = _find_auto_project_root(cur)
    if auto_root != cur and cur.is_relative_to(auto_root):
        fallback: list[Path] = []
        for candidate in (cur, *cur.parents):
            fallback.append(candidate)
            if candidate == auto_root:
                break
        return tuple(fallback)
    return (cur,)


def find_project_root(
    start: Path | None = None,
    *,
    explicit_root: Path | None = None,
) -> Path:
    """Resolve project root from explicit path, PROBE_PROJECT_ROOT, or nearest marker."""
    if explicit_root is not None:
        p = Path(explicit_root).expanduser()
        resolved = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
        if not resolved.is_dir():
            raise ValueError(f"Project root directory does not exist: {explicit_root}")
        return resolved

    if "PROBE_PROJECT_ROOT" in os.environ:
        raw_env = os.environ["PROBE_PROJECT_ROOT"].strip()
        if raw_env:
            p = Path(raw_env).expanduser()
            if p.is_absolute():
                resolved = p.resolve()
            else:
                cwd_cand = (Path.cwd() / p).resolve()
                start_cand = (start.resolve() / p).resolve() if start is not None else cwd_cand
                resolved = cwd_cand if cwd_cand.is_dir() or start is None else start_cand
            if not resolved.is_dir():
                raise ValueError(f"PROBE_PROJECT_ROOT directory does not exist: {raw_env}")
            return resolved
        return _find_auto_project_root(start)

    cur = (start or Path.cwd()).resolve()
    auto_root = _find_auto_project_root(cur)
    for env_dir in find_env_search_dirs(auto_root):
        raw_val = _read_dotenv_cached(env_dir / ".env", "PROBE_PROJECT_ROOT")
        if raw_val is not None:
            if not raw_val:
                break
            p = Path(raw_val).expanduser()
            resolved = p.resolve() if p.is_absolute() else (env_dir / p).resolve()
            if not resolved.is_dir():
                raise ValueError(f"PROBE_PROJECT_ROOT directory does not exist: {raw_val}")
            return resolved
    return auto_root


def _find_pyproject(start: Path) -> Path | None:
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            return pyproject
        if (candidate / ".git").exists():
            break
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _parse_bool_config(key: str, val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        norm = val.strip().lower()
        if norm in _TRUE_VALUES:
            return True
        if norm in _FALSE_VALUES:
            return False
    raise ValueError(f"Config key {key!r} must be a boolean, got {val!r}")


def _apply_dataclass_dict(obj: Any, data: dict[str, Any]) -> None:
    for key, val in data.items():
        if key in {"env_search_dirs", "env_files"} and isinstance(obj, ModelEndpointConfig):
            continue
        if key == "db_path" and isinstance(obj, RuntimeConfig):
            continue
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key {key!r} in {type(obj).__name__}")
        current = getattr(obj, key)
        if key in _OPTIONAL_INT_FIELDS and isinstance(obj, ModelEndpointConfig):
            if val is None:
                setattr(obj, key, None)
            elif isinstance(val, (bool, float)):
                raise ValueError(f"Config key {key!r} must be an integer, got {type(val).__name__} {val!r}")
            else:
                try:
                    setattr(obj, key, int(val))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Config key {key!r} must be an integer, got {val!r}") from exc
        elif isinstance(current, tuple):
            if isinstance(val, str):
                items: Any = [val]
            elif isinstance(val, (list, tuple)):
                items = val
            else:
                raise ValueError(f"Config key {key!r} must be a list of strings, got {type(val).__name__}")
            setattr(obj, key, tuple(str(x) for x in items))
        elif isinstance(current, bool):
            setattr(obj, key, _parse_bool_config(key, val))
        elif isinstance(current, int):
            if isinstance(val, (bool, float)):
                raise ValueError(f"Config key {key!r} must be an integer, got {type(val).__name__} {val!r}")
            try:
                setattr(obj, key, int(val))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Config key {key!r} must be an integer, got {val!r}") from exc
        elif isinstance(current, float):
            if isinstance(val, bool):
                raise ValueError(f"Config key {key!r} must be a float, got bool {val!r}")
            try:
                fval = float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Config key {key!r} must be a float, got {val!r}") from exc
            if not math.isfinite(fval):
                raise ValueError(f"Config key {key!r} must be finite, got {val!r}")
            setattr(obj, key, fval)
        else:
            setattr(obj, key, str(val))


def _url_origin(url: str) -> tuple[str, str, str]:
    stripped = (url or "").strip()
    if not stripped:
        return ("", "", "")
    try:
        parsed = urlparse(stripped)
    except ValueError as exc:
        raise ValueError(f"Invalid endpoint URL {url!r}: {exc}") from exc
    if parsed.scheme and not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL {url!r}: missing host")
    return (parsed.scheme.lower(), parsed.netloc.lower(), (parsed.hostname or "").lower())


def _merge_endpoint_config(
    default_ep: ModelEndpointConfig,
    user_dict: dict[str, Any],
    proj_dict: dict[str, Any],
) -> ModelEndpointConfig:
    """Merge user and project endpoint dicts while isolating credentials across provider/origin swaps."""
    ep = replace(default_ep)
    user_host = _url_origin(str(user_dict["url"]))[2] if "url" in user_dict else ""

    for is_proj, layer in ((False, user_dict), (True, proj_dict)):
        if not layer:
            continue
        new_provider = str(layer["provider"]).strip().lower() if "provider" in layer else None
        prev_provider = ep.provider.strip().lower()
        provider_changed = new_provider is not None and new_provider != prev_provider

        new_url = str(layer["url"]).strip() if "url" in layer else None
        prev_origin = _url_origin(ep.url)[:2]
        new_origin = _url_origin(new_url)[:2] if new_url is not None else prev_origin
        url_changed = new_url is not None and new_origin != prev_origin

        if provider_changed or url_changed:
            if "api_key" not in layer:
                ep.api_key = ""
            if "api_key_env" not in layer:
                ep.api_key_env = ""
            if "reasoning_effort" not in layer:
                ep.reasoning_effort = ""
            eff_provider = new_provider if new_provider is not None else prev_provider
            if "model" not in layer and not (not provider_changed and eff_provider == "typesafe_jev"):
                ep.model = ""
            if provider_changed and "url" not in layer:
                ep.url = ""
        elif "api_key_env" in layer and "api_key" not in layer:
            ep.api_key = ""

        _apply_dataclass_dict(ep, layer)

        if is_proj and "url" in layer:
            _, _, proj_hostname = _url_origin(ep.url)
            trusted_hosts = _TRUSTED_DEFAULT_HOSTS | _LOOPBACK_HOSTS | ({user_host} if user_host else set())
            if (
                proj_hostname
                and proj_hostname not in trusted_hosts
                and (ep.api_key or ep.api_key_env)
                and os.environ.get("PROBE_TRUST_PROJECT_ENDPOINTS", "").strip().lower() not in _TRUE_VALUES
            ):
                raise ValueError(
                    f"Refusing to attach credentials (api_key/api_key_env) to project-defined "
                    f"endpoint host {proj_hostname!r} in pyproject.toml; configure it in "
                    "~/.config/probe/config.toml or set PROBE_TRUST_PROJECT_ENDPOINTS=1."
                )
    return ep


def _load_toml_file(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc


def _validate_top_level_keys(data: dict[str, Any], source_label: str) -> None:
    allowed_top = {"models", "decision", "text", "rate_limit", "check", "redact", "runtime", "domain_context"}
    for k in data:
        if k not in allowed_top:
            raise ValueError(f"Unknown config section/key {k!r} in {source_label}")
    models = data.get("models")
    if models is not None:
        if not isinstance(models, dict):
            raise ValueError(f"[models] must be a table in {source_label}")
        for mk in models:
            if mk not in {"decision", "text"}:
                raise ValueError(f"Unknown [models.{mk}] section in {source_label}")


def load_config(
    cwd: Path | None = None,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> ProbeConfig:
    """Load default config merged with ~/.config/probe/config.toml (or PROBE_CONFIG) and pyproject.toml."""
    work_dir = (cwd or Path.cwd()).resolve()
    resolved_root = find_project_root(work_dir, explicit_root=project_root)
    cfg = ProbeConfig(project_root=resolved_root)

    raw_user: dict[str, Any] = {}
    env_cfg = _read_env_flag("PROBE_CONFIG", resolved_root)
    user_path = config_path or (Path(env_cfg).expanduser() if env_cfg else _DEFAULT_USER_CONFIG_PATH)
    if user_path.is_file():
        raw_user = _load_toml_file(user_path)
        _validate_top_level_keys(raw_user, str(user_path))

    raw_proj: dict[str, Any] = {}
    pyproject = _find_pyproject(resolved_root)
    if pyproject is not None:
        parsed = _load_toml_file(pyproject)
        tool_probe = parsed.get("tool", {}).get("probe")
        if isinstance(tool_probe, dict):
            _validate_top_level_keys(tool_probe, str(pyproject))
            raw_proj = tool_probe

    user_models = raw_user.get("models") if isinstance(raw_user.get("models"), dict) else {}
    proj_models = raw_proj.get("models") if isinstance(raw_proj.get("models"), dict) else {}
    user_dec = user_models.get("decision") if isinstance(user_models.get("decision"), dict) else {}
    proj_dec = proj_models.get("decision") if isinstance(proj_models.get("decision"), dict) else {}
    user_txt = user_models.get("text") if isinstance(user_models.get("text"), dict) else {}
    proj_txt = proj_models.get("text") if isinstance(proj_models.get("text"), dict) else {}

    cfg.models_decision = _merge_endpoint_config(cfg.models_decision, user_dec, proj_dec)
    cfg.models_text = _merge_endpoint_config(cfg.models_text, user_txt, proj_txt)

    merged = _deep_merge(raw_user, raw_proj)
    for section_name, target in (
        ("decision", cfg.decision),
        ("text", cfg.text),
        ("rate_limit", cfg.rate_limit),
        ("check", cfg.check),
        ("redact", cfg.redact),
        ("runtime", cfg.runtime),
    ):
        section = merged.get(section_name)
        if section is not None:
            if not isinstance(section, dict):
                raise ValueError(f"[{section_name}] must be a TOML table")
            _apply_dataclass_dict(target, section)

    if "domain_context" in merged and merged["domain_context"] is not None:
        if not isinstance(merged["domain_context"], str):
            raise ValueError(f"domain_context must be a string, got {type(merged['domain_context']).__name__}")
        cfg.domain_context = merged["domain_context"].strip()

    env_search_dirs = find_env_search_dirs(resolved_root)
    cfg.models_decision.env_search_dirs = env_search_dirs
    cfg.models_decision.env_files = cfg.redact.env_files
    cfg.models_text.env_search_dirs = env_search_dirs
    cfg.models_text.env_files = cfg.redact.env_files

    for role_name, ep in (("models.decision", cfg.models_decision), ("models.text", cfg.models_text)):
        if not math.isfinite(ep.timeout_s) or ep.timeout_s <= 0.0:
            raise ValueError(f"[{role_name}].timeout_s must be > 0, got {ep.timeout_s!r}")
        if ep.max_length < 8:
            raise ValueError(f"[{role_name}].max_length must be >= 8, got {ep.max_length!r}")
        if ep.onnx_output.strip().lower() not in _ALLOWED_ONNX_OUTPUTS:
            raise ValueError(
                f"[{role_name}].onnx_output must be 'logits' or 'probabilities', got {ep.onnx_output!r}"
            )
        _url_origin(ep.url)

    if cfg.text.detail.strip().lower() not in _ALLOWED_TEXT_DETAILS:
        raise ValueError(f"[text].detail must be concise, standard, or detailed; got {cfg.text.detail!r}")
    if cfg.text.max_tokens < 1:
        raise ValueError(f"[text].max_tokens must be >= 1, got {cfg.text.max_tokens!r}")
    if not math.isfinite(cfg.decision.pass_threshold) or not (0.0 <= cfg.decision.pass_threshold < 1.0):
        raise ValueError(f"[decision].pass_threshold must be in [0.0, 1.0), got {cfg.decision.pass_threshold!r}")
    if cfg.rate_limit.max_calls < 1:
        raise ValueError(f"[rate_limit].max_calls must be >= 1, got {cfg.rate_limit.max_calls!r}")
    if not math.isfinite(cfg.rate_limit.window_seconds) or cfg.rate_limit.window_seconds <= 0.0:
        raise ValueError(f"[rate_limit].window_seconds must be > 0, got {cfg.rate_limit.window_seconds!r}")
    if cfg.rate_limit.max_calls_per_run < -1:
        raise ValueError(
            f"[rate_limit].max_calls_per_run must be >= -1, got {cfg.rate_limit.max_calls_per_run!r}"
        )
    if cfg.check.max_checks < 1 or cfg.check.max_check_chars < 1 or cfg.check.max_workers < 1:
        raise ValueError(
            "Invalid [check] bounds: max_checks >= 1, max_check_chars >= 1, and max_workers >= 1 required"
        )
    if (
        cfg.runtime.max_depth < 1
        or cfg.runtime.max_items < 1
        or cfg.runtime.max_str_bytes < 64
        or cfg.runtime.max_workers < 1
    ):
        raise ValueError(
            "Invalid [runtime] bounds: max_depth >= 1, max_items >= 1, "
            "max_str_bytes >= 64, and max_workers >= 1 required"
        )

    return cfg
