"""Configuration loading, environment helpers, and ID/timestamp utilities."""

from __future__ import annotations

import math
import os
import threading
import time
import tomllib
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
_EXPORT_PREFIX_RE = ("export ", "export\t")


def _strip_inline_comment(raw: str) -> str:
    for idx, ch in enumerate(raw):
        if ch == "#" and idx > 0 and raw[idx - 1].isspace():
            return raw[:idx].rstrip()
    return raw.rstrip()


def _parse_dotenv_key(env_path: Path, key_name: str) -> str:
    """Return value for key_name from a dotenv file, or empty string if absent."""
    try:
        text = env_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for prefix in _EXPORT_PREFIX_RE:
            if line.startswith(prefix):
                line = line[len(prefix) :].lstrip()
                break
        if "=" not in line:
            continue
        k, _, raw_val = line.partition("=")
        if k.strip() != key_name:
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
                return "".join(chars).strip()
            end_idx = val.find("'", 1)
            return (val[1:end_idx] if end_idx != -1 else val[1:]).strip()
        return _strip_inline_comment(val).strip()
    return ""


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


_ENV_FLAG_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_ENV_FLAG_CACHE_LOCK = threading.Lock()
_ENV_FLAG_TTL_S = 0.5
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def clear_env_cache() -> None:
    """Clear cached .env flag lookups (used after environment or directory changes in tests)."""
    with _ENV_FLAG_CACHE_LOCK:
        _ENV_FLAG_CACHE.clear()


def _read_env_flag(var_name: str, cwd: Path | None = None) -> str:
    if var_name in os.environ:
        return os.environ[var_name].strip()

    base_str = str(cwd) if cwd is not None else os.getcwd()
    cache_key = (base_str, var_name)
    now = time.monotonic()
    with _ENV_FLAG_CACHE_LOCK:
        cached = _ENV_FLAG_CACHE.get(cache_key)
        if cached is not None and now < cached[0]:
            return cached[1]

    root = find_project_root(Path(base_str))
    env_path = root / ".env"
    parsed = _parse_dotenv_key(env_path, var_name) if env_path.is_file() else ""
    with _ENV_FLAG_CACHE_LOCK:
        _ENV_FLAG_CACHE[cache_key] = (now + _ENV_FLAG_TTL_S, parsed)
    return parsed


def is_probes_enabled(cwd: Path | None = None) -> bool:
    """Return True when PROBES is enabled in os.environ or project .env."""
    return _read_env_flag("PROBES", cwd).lower() in _TRUE_VALUES


def is_skip_lock(cwd: Path | None = None) -> bool:
    """Return True when PROBE_SKIP_LOCK=1 bypasses .probes/lock.json checks."""
    return _read_env_flag("PROBE_SKIP_LOCK", cwd).lower() in _TRUE_VALUES


def is_no_pdb(cwd: Path | None = None) -> bool:
    """Return True when PROBE_NO_PDB=1 disables interactive pdb halts."""
    return _read_env_flag("PROBE_NO_PDB", cwd).lower() in _TRUE_VALUES


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
    entailment_index: int = 1
    contradiction_index: int = 0
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
        env_val = os.environ.get(self.api_key_env, "").strip()
        if env_val:
            return env_val

        seen: set[Path] = set()
        for base_dir in self.env_search_dirs:
            for raw_name in self.env_files:
                candidate = Path(raw_name).expanduser()
                if not candidate.is_absolute():
                    candidate = base_dir / candidate
                resolved = candidate.resolve()
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
    max_tokens: int = 400
    explain_decision_failures: bool = True
    system_prompt: str = ""


@dataclass(slots=True)
class RateLimitConfig:
    """Per-probe sliding-window and optional per-run call caps (-1 = unlimited)."""

    max_calls: int = 5
    window_seconds: float = 10.0
    max_calls_per_run: int = -1


@dataclass(slots=True)
class CheckConfig:
    """Pre-run probe check budgets and lockfile enforcement."""

    max_checks: int = 5
    max_check_chars: int = 200
    require_check: bool = True


@dataclass(slots=True)
class RedactConfig:
    """Secret redaction sources and extra field names."""

    env_files: tuple[str, ...] = (".env",)
    extra_keys: tuple[str, ...] = ()


@dataclass(slots=True)
class RuntimeConfig:
    """Storage paths, snapshot caps, and strict API handling."""

    db_path: str = ".probes/probes.db"
    lock_path: str = ".probes/lock.json"
    strict_api: bool = False
    max_depth: int = 4
    max_items: int = 25
    max_str_bytes: int = 4096


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
    project_root: Path = field(default_factory=Path.cwd)

    def resolve_db_path(self) -> Path:
        path = Path(self.runtime.db_path)
        return path if path.is_absolute() else (self.project_root / path)

    def resolve_lock_path(self) -> Path:
        path = Path(self.runtime.lock_path)
        return path if path.is_absolute() else (self.project_root / path)


def find_project_root(start: Path | None = None) -> Path:
    """Locate nearest directory with pyproject.toml, .probes, or .git; fallback to start."""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".probes").is_dir():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return cur


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
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(current, tuple):
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
            if isinstance(val, bool):
                raise ValueError(f"Config key {key!r} must be an integer, got bool {val!r}")
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


def _merge_endpoint_config(
    default_ep: ModelEndpointConfig,
    user_dict: dict[str, Any],
    proj_dict: dict[str, Any],
) -> ModelEndpointConfig:
    """Merge user and project endpoint dicts while isolating credentials across provider/host swaps."""
    ep = replace(default_ep)
    for layer in (user_dict, proj_dict):
        if not layer:
            continue
        new_provider = str(layer["provider"]).strip().lower() if "provider" in layer else None
        prev_provider = ep.provider.strip().lower()
        provider_changed = new_provider is not None and new_provider != prev_provider

        new_url = str(layer["url"]).strip() if "url" in layer else None
        prev_host = urlparse(ep.url).netloc.lower() if ep.url else ""
        new_host = urlparse(new_url).netloc.lower() if new_url else ""
        url_changed = new_url is not None and new_host != prev_host

        if provider_changed or url_changed:
            if "api_key" not in layer and "api_key_env" not in layer:
                ep.api_key = ""
                ep.api_key_env = ""
            if provider_changed and "url" not in layer:
                ep.url = ""
            if provider_changed and "model" not in layer:
                ep.model = ""

        _apply_dataclass_dict(ep, layer)
    return ep


def _load_toml_file(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc


def load_config(cwd: Path | None = None, config_path: Path | None = None) -> ProbeConfig:
    """Load default config merged with ~/.config/probe/config.toml (or PROBE_CONFIG) and pyproject.toml."""
    work_dir = (cwd or Path.cwd()).resolve()
    project_root = find_project_root(work_dir)
    cfg = ProbeConfig(project_root=project_root)

    raw_user: dict[str, Any] = {}
    env_cfg = _read_env_flag("PROBE_CONFIG", work_dir)
    user_path = config_path or (Path(env_cfg).expanduser() if env_cfg else _DEFAULT_USER_CONFIG_PATH)
    if user_path.is_file():
        raw_user = _load_toml_file(user_path)

    raw_proj: dict[str, Any] = {}
    pyproject = _find_pyproject(work_dir)
    if pyproject is not None:
        parsed = _load_toml_file(pyproject)
        tool_probe = parsed.get("tool", {}).get("probe")
        if isinstance(tool_probe, dict):
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
        if isinstance(section, dict):
            _apply_dataclass_dict(target, section)

    cfg.models_decision.env_search_dirs = (project_root,)
    cfg.models_decision.env_files = cfg.redact.env_files
    cfg.models_text.env_search_dirs = (project_root,)
    cfg.models_text.env_files = cfg.redact.env_files

    if not math.isfinite(cfg.decision.pass_threshold) or not (0.0 <= cfg.decision.pass_threshold < 1.0):
        raise ValueError(f"[decision].pass_threshold must be in [0.0, 1.0), got {cfg.decision.pass_threshold!r}")
    if cfg.rate_limit.max_calls < 1:
        raise ValueError(f"[rate_limit].max_calls must be >= 1, got {cfg.rate_limit.max_calls!r}")
    if not math.isfinite(cfg.rate_limit.window_seconds) or cfg.rate_limit.window_seconds <= 0.0:
        raise ValueError(f"[rate_limit].window_seconds must be > 0, got {cfg.rate_limit.window_seconds!r}")
    if cfg.runtime.max_depth < 1 or cfg.runtime.max_items < 1 or cfg.runtime.max_str_bytes < 64:
        raise ValueError("Invalid [runtime] bounds: max_depth >= 1, max_items >= 1, max_str_bytes >= 64 required")

    return cfg
