"""Configuration loading, environment helpers, and ID/timestamp utilities."""

from __future__ import annotations

import os
import threading
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


_ENV_FLAG_CACHE: dict[tuple[Path, str], tuple[int, str]] = {}
_ENV_FLAG_CACHE_LOCK = threading.Lock()


def _read_env_flag(var_name: str, cwd: Path | None = None) -> str:
    env_val = os.environ.get(var_name)
    if env_val is not None and env_val.strip() != "":
        return env_val.strip()

    root = find_project_root(cwd)
    env_path = (root / ".env").resolve()
    try:
        mtime_ns = env_path.stat().st_mtime_ns
    except OSError:
        return ""

    cache_key = (env_path, var_name)
    with _ENV_FLAG_CACHE_LOCK:
        cached = _ENV_FLAG_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]
        parsed = _parse_dotenv_key(env_path, var_name)
        _ENV_FLAG_CACHE[cache_key] = (mtime_ns, parsed)
        return parsed


def is_probes_enabled(cwd: Path | None = None) -> bool:
    """Return True when PROBES is enabled in os.environ or project .env."""
    return _read_env_flag("PROBES", cwd).lower() in {"1", "true", "yes", "on"}


def is_skip_lock(cwd: Path | None = None) -> bool:
    """Return True when PROBE_SKIP_LOCK=1 bypasses .probes/lock.json checks."""
    return _read_env_flag("PROBE_SKIP_LOCK", cwd).lower() in {"1", "true", "yes", "on"}


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
    max_length: int = 256
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


def _apply_dataclass_dict(obj: Any, data: dict[str, Any]) -> None:
    for key, val in data.items():
        if key in {"env_search_dirs", "env_files"} and isinstance(obj, ModelEndpointConfig):
            continue
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(current, tuple):
            items = [val] if isinstance(val, str) else val
            setattr(obj, key, tuple(str(x) for x in items))
        elif isinstance(current, bool):
            setattr(obj, key, bool(val))
        elif isinstance(current, int):
            setattr(obj, key, int(val))
        elif isinstance(current, float):
            setattr(obj, key, float(val))
        else:
            setattr(obj, key, str(val))


def load_config(cwd: Path | None = None, config_path: Path | None = None) -> ProbeConfig:
    """Load default config merged with ~/.config/probe/config.toml (or PROBE_CONFIG) and pyproject.toml."""
    work_dir = (cwd or Path.cwd()).resolve()
    project_root = find_project_root(work_dir)
    cfg = ProbeConfig(project_root=project_root)

    raw_user: dict[str, Any] = {}
    env_cfg = os.environ.get("PROBE_CONFIG", "").strip()
    user_path = config_path or (Path(env_cfg).expanduser() if env_cfg else _DEFAULT_USER_CONFIG_PATH)
    if user_path.is_file():
        raw_user = tomllib.loads(user_path.read_text(encoding="utf-8"))

    raw_proj: dict[str, Any] = {}
    pyproject = _find_pyproject(work_dir)
    if pyproject is not None:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        tool_probe = parsed.get("tool", {}).get("probe")
        if isinstance(tool_probe, dict):
            raw_proj = tool_probe

    merged = _deep_merge(raw_user, raw_proj)

    models_raw = merged.get("models", {})
    explicit_text_cfg = False
    if isinstance(models_raw, dict):
        if isinstance(models_raw.get("decision"), dict):
            _apply_dataclass_dict(cfg.models_decision, models_raw["decision"])
        if isinstance(models_raw.get("text"), dict):
            _apply_dataclass_dict(cfg.models_text, models_raw["text"])
            explicit_text_cfg = True

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

    if not explicit_text_cfg and not cfg.models_text.resolve_api_key():
        gemini_candidate = ModelEndpointConfig(
            provider="openai_compat",
            url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-3.8-flash",
            api_key_env="GEMINI_API_KEY",
            reasoning_effort="none",
            env_search_dirs=(project_root,),
            env_files=cfg.redact.env_files,
        )
        if gemini_candidate.resolve_api_key():
            cfg.models_text = gemini_candidate
    return cfg
