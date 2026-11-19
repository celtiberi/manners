"""Decision and text model backends for runtime grading and pre-run checks."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from probe.config import DecisionConfig, ModelEndpointConfig, TextConfig
from probe.prompts import (
    compose_system_prompt,
    format_analyze_snapshot_prompts,
    format_explain_failure_prompts,
    format_gate2_analysis_prompts,
    format_gate3_fixtures_prompts,
    format_typesafe_state,
    get_output_schema,
)

_THINK_BLOCK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"\A\s*<think>[\s\S]*\Z", re.DOTALL | re.IGNORECASE)
_OUTER_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*)\s*```", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_DEFAULT_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_DEFAULT_TYPESAFE_MODEL = "jev-latest"
_LEGACY_JEV_MODELS = frozenset({"", "jev", "typesafe-ai/jev"})
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
_MAX_HTTP_ATTEMPTS = 2
_RETRY_BACKOFF_S = 0.4
_USER_AGENT = "probe/0.6"


class BackendError(RuntimeError):
    """Raised when a configured decision or text model backend fails or returns invalid data."""


class ModelTruncatedError(BackendError):
    """Raised when a text model hits max_tokens (finish_reason='length') before completing."""

    def __init__(self, message: str, *, partial_text: str = "") -> None:
        super().__init__(message)
        self.partial_text = partial_text


@dataclass(slots=True, frozen=True)
class CheckScore:
    """Calibrated binary verdict for a single discrete check."""

    check: str
    probability: float
    passed: bool


def strip_thinking(text: str) -> str:
    """Remove leading <think>...</think> reasoning blocks emitted by thinking models."""
    cleaned = text or ""
    while True:
        updated = _THINK_BLOCK_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text, ignoring leading <think> tags and fences."""
    cleaned = strip_thinking(text)
    outer_fence = _OUTER_CODE_FENCE_RE.search(cleaned)
    fenced = [match.group(1).strip() for match in _CODE_FENCE_RE.finditer(cleaned)]
    candidates: list[str] = []
    if outer_fence is not None:
        candidates.append(outer_fence.group(1).strip())
    candidates.extend(fenced)
    candidates.append(cleaned)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: locate balanced outer braces inside prose (stop if an outer brace is unclosed).
        start = candidate.find("{")
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            closed = False
            for idx in range(start, len(candidate)):
                ch = candidate[idx]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        closed = True
                        snippet = candidate[start : idx + 1]
                        try:
                            parsed = json.loads(snippet)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
            if not closed:
                break
            start = candidate.find("{", start + 1)

    raise BackendError(f"Expected JSON object in model response, got: {cleaned[:200]!r}")


def _clamp_prob(value: Any) -> float:
    if isinstance(value, bool):
        raise BackendError(f"Probability must be numeric, got bool {value!r}")
    try:
        p = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"Invalid probability value: {value!r}") from exc
    if not math.isfinite(p):
        raise BackendError(f"Non-finite probability value: {value!r}")
    if p < -0.05 or p > 1.05:
        raise BackendError(f"Probability out of [0, 1] bounds: {value!r}")
    return max(0.0, min(1.0, p))


def _chat_completions_url(base_url: str) -> str:
    stripped = (base_url or "").strip().rstrip("/")
    if not stripped:
        raise BackendError("Model URL is empty")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _auth_headers(endpoint: ModelEndpointConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    api_key = endpoint.resolve_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _is_local_ollama_url(url_lower: str) -> bool:
    return ":11434" in url_lower or "ollama" in url_lower


def _apply_reasoning_effort(
    payload: dict[str, Any],
    endpoint: ModelEndpointConfig,
    url: str,
) -> None:
    effort = (endpoint.reasoning_effort or "").strip()
    if effort.lower() == "off":
        return
    url_lower = url.lower()
    model_lower = (endpoint.model or "").lower()
    if not effort and "qwen3" in model_lower and _is_local_ollama_url(url_lower):
        effort = "none"
    if effort:
        payload["reasoning_effort"] = effort


def _schema_has_open_object(node: Any) -> bool:
    """Return True if any JSON Schema object node omits non-empty properties."""
    if isinstance(node, list):
        return any(_schema_has_open_object(item) for item in node)
    if not isinstance(node, dict):
        return False

    node_type = node.get("type")
    is_obj = node_type == "object" or (isinstance(node_type, list) and "object" in node_type)
    props = node.get("properties")
    if is_obj and (not isinstance(props, dict) or not props):
        return True

    for key in ("properties", "$defs", "definitions"):
        sub = node.get(key)
        if isinstance(sub, dict) and any(_schema_has_open_object(v) for v in sub.values()):
            return True
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), (dict, list)) and _schema_has_open_object(node[key]):
            return True
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(node.get(key), list) and _schema_has_open_object(node[key]):
            return True
    return False


def _schema_is_openai_strict_compatible(node: Any) -> bool:
    """Return True if every object node defines properties, all-required keys, and additionalProperties=False."""
    if isinstance(node, list):
        return all(_schema_is_openai_strict_compatible(item) for item in node)
    if not isinstance(node, dict):
        return True

    node_type = node.get("type")
    is_obj = node_type == "object" or (isinstance(node_type, list) and "object" in node_type)
    if is_obj:
        props = node.get("properties")
        req = node.get("required")
        if (
            not isinstance(props, dict)
            or not props
            or node.get("additionalProperties") is not False
            or not isinstance(req, list)
            or set(req) != set(props.keys())
        ):
            return False

    for key in ("properties", "$defs", "definitions"):
        sub = node.get(key)
        if isinstance(sub, dict) and not all(_schema_is_openai_strict_compatible(v) for v in sub.values()):
            return False
    if isinstance(node.get("items"), (dict, list)) and not _schema_is_openai_strict_compatible(node["items"]):
        return False
    if isinstance(node.get("additionalProperties"), (dict, list)) and not _schema_is_openai_strict_compatible(
        node["additionalProperties"]
    ):
        return False
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(node.get(key), list) and not all(
            _schema_is_openai_strict_compatible(v) for v in node[key]
        ):
            return False
    return True


def _strip_additional_properties(node: Any) -> Any:
    """Return a copy of a JSON Schema with 'additionalProperties' keywords stripped."""
    if isinstance(node, list):
        return [_strip_additional_properties(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, val in node.items():
        if key == "additionalProperties":
            continue
        if key in ("properties", "$defs", "definitions") and isinstance(val, dict):
            out[key] = {k: _strip_additional_properties(v) for k, v in val.items()}
        elif key == "items" and isinstance(val, (dict, list)):
            out[key] = _strip_additional_properties(val)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(val, list):
            out[key] = [_strip_additional_properties(v) for v in val]
        else:
            out[key] = val
    return out


def _apply_output_schema(
    payload: dict[str, Any],
    system_text: str,
    output_schema: dict[str, Any] | None,
    url: str,
    model: str = "",
) -> str:
    """Translate declarative output_schema to the target OpenAI-compat provider format."""
    if output_schema is None:
        return system_text
    name = str(output_schema.get("name") or "response")
    schema = output_schema["schema"]
    url_lower = url.lower()
    model_lower = (model or "").lower()
    is_deepseek = "deepseek" in url_lower or "deepseek" in model_lower
    is_gemini = "generativelanguage.googleapis.com" in url_lower
    is_openai = "api.openai.com" in url_lower

    if is_deepseek:
        schema_json = json.dumps(schema, ensure_ascii=False)
        if "reasoner" not in model_lower:
            payload["response_format"] = {"type": "json_object"}
        return f"{system_text}\n\nRespond with a JSON object matching this JSON Schema:\n{schema_json}"

    if (is_gemini or is_openai) and _schema_has_open_object(schema):
        payload["response_format"] = {"type": "json_object"}
        schema_json = json.dumps(schema, ensure_ascii=False)
        return f"{system_text}\n\nRespond with a JSON object matching this JSON Schema:\n{schema_json}"

    effective_schema = _strip_additional_properties(schema) if is_gemini else schema
    json_schema_spec: dict[str, Any] = {"name": name, "schema": effective_schema}
    if is_openai and _schema_is_openai_strict_compatible(effective_schema):
        json_schema_spec["strict"] = True
    payload["response_format"] = {"type": "json_schema", "json_schema": json_schema_spec}
    return system_text


def _post_json_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    endpoint: ModelEndpointConfig,
    *,
    role_label: str,
) -> dict[str, Any]:
    headers = _auth_headers(endpoint)
    for attempt in range(1, _MAX_HTTP_ATTEMPTS + 1):
        try:
            resp = client.post(url, json=payload, headers=headers)
        except httpx.InvalidURL as exc:
            raise BackendError(f"{role_label} invalid URL ({url}): {exc}") from exc
        except RuntimeError as exc:
            raise BackendError(f"{role_label} HTTP client is closed ({url}): {exc}") from exc
        except httpx.HTTPError as exc:
            if attempt < _MAX_HTTP_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S)
                continue
            raise BackendError(f"{role_label} HTTP request failed ({url}): {exc}") from exc

        if resp.status_code in _RETRY_STATUS_CODES and attempt < _MAX_HTTP_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_S)
            continue

        try:
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            snippet = resp.text[:300].strip()
            detail = f" — {snippet}" if snippet else ""
            raise BackendError(f"{role_label} HTTP {resp.status_code} ({url}){detail}") from exc
        except ValueError as exc:
            raise BackendError(f"{role_label} endpoint returned invalid JSON ({url}): {exc}") from exc

        if not isinstance(data, dict):
            raise BackendError(f"{role_label} endpoint returned non-object JSON: {type(data).__name__}")
        return data

    raise BackendError(f"{role_label} request failed after {_MAX_HTTP_ATTEMPTS} attempts ({url})")


class DecisionBackend:
    """Evaluates discrete checks against a snapshot and returns CheckScore values."""

    def __init__(
        self,
        endpoint: ModelEndpointConfig,
        decision_cfg: DecisionConfig | None = None,
        *,
        domain_context: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.pass_threshold = (decision_cfg or DecisionConfig()).pass_threshold
        self.domain_context = domain_context.strip()
        self._client = httpx.Client(
            timeout=httpx.Timeout(endpoint.timeout_s, connect=10.0),
            transport=transport,
        )
        self._onnx_session: Any = None
        self._onnx_tokenizer: Any = None
        self._onnx_np: Any = None
        self._onnx_input_names: set[str] = set()
        self._onnx_ent_idx: int | None = None
        self._onnx_con_idx: int | None = None
        self._onnx_lock = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def evaluate_checks(
        self,
        *,
        probe_id: str,
        checks: Sequence[str],
        snapshot_json: str,
        analysis: str = "",
    ) -> list[CheckScore]:
        """Score every check in [0.0, 1.0] using the configured decision provider."""
        check_list = list(checks)
        if not check_list:
            return []

        provider = self.endpoint.provider.strip().lower()
        if provider == "typesafe_jev":
            probs = self._eval_typesafe_jev(probe_id, check_list, snapshot_json, analysis)
        elif provider == "http_decision":
            probs = self._eval_http_decision(probe_id, check_list, snapshot_json, analysis)
        elif provider == "onnx_nli":
            probs = self._eval_onnx_nli(check_list, snapshot_json, analysis)
        else:
            raise BackendError(
                f"Unsupported decision provider {self.endpoint.provider!r}: "
                "[models.decision] requires a decision model ('typesafe_jev', 'http_decision', or 'onnx_nli'), "
                "not a text generation LLM."
            )

        return [
            CheckScore(
                check=check,
                probability=round(p, 4),
                passed=p > self.pass_threshold,
            )
            for check, p in zip(check_list, probs, strict=True)
        ]

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json_with_retry(
            self._client,
            url,
            payload,
            self.endpoint,
            role_label="Decision",
        )

    def _eval_typesafe_jev(
        self,
        probe_id: str,
        checks: list[str],
        snapshot_json: str,
        analysis: str,
    ) -> list[float]:
        url = (self.endpoint.url or "").strip() or _DEFAULT_TYPESAFE_URL
        if "typesafe.ai" in url.lower() and not self.endpoint.resolve_api_key():
            key_name = self.endpoint.api_key_env or "TYPESAFE_API_KEY"
            raise BackendError(f"Missing TypeSafe API key ({key_name})")

        raw_model = (self.endpoint.model or "").strip()
        eff_model = _DEFAULT_TYPESAFE_MODEL if raw_model in _LEGACY_JEV_MODELS else raw_model

        state_str = format_typesafe_state(
            probe_id,
            snapshot_json,
            analysis,
            domain_context=self.domain_context,
        )

        questions = {
            f"q{i}": {"type": "noul", "instructions": check}
            for i, check in enumerate(checks)
        }
        payload = {
            "model": eff_model,
            "state": state_str,
            "questions": questions,
        }
        data = self._post_json(url, payload)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise BackendError(f"TypeSafe response missing 'answers' object: {data!r}")

        probs: list[float] = []
        for i, _check in enumerate(checks):
            q_key = f"q{i}"
            entry = answers.get(q_key)
            if not isinstance(entry, dict) or "noul" not in entry:
                raise BackendError(f"TypeSafe response missing noul answer for {q_key}: {data!r}")
            raw_val = entry["noul"]
            if isinstance(raw_val, bool):
                raise BackendError(f"TypeSafe noul value for {q_key} must be float, got bool {raw_val!r}")
            probs.append(_clamp_prob(raw_val))
        return probs

    def _eval_http_decision(
        self,
        probe_id: str,
        checks: list[str],
        snapshot_json: str,
        analysis: str,
    ) -> list[float]:
        url = (self.endpoint.url or "").strip()
        if not url:
            raise BackendError("http_decision provider requires non-empty `url` in [models.decision]")
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "probe_id": probe_id,
            "checks": checks,
            "snapshot": snapshot_json,
        }
        if analysis:
            payload["analysis"] = analysis
        if self.domain_context:
            payload["domain_context"] = self.domain_context

        data = self._post_json(url, payload)
        return _parse_probabilities_payload(data, checks)

    def _ensure_onnx_loaded(self) -> tuple[Any, Any, Any]:
        if self._onnx_session is not None and self._onnx_tokenizer is not None and self._onnx_np is not None:
            return self._onnx_session, self._onnx_tokenizer, self._onnx_np
        with self._onnx_lock:
            if (
                self._onnx_session is not None
                and self._onnx_tokenizer is not None
                and self._onnx_np is not None
            ):
                return self._onnx_session, self._onnx_tokenizer, self._onnx_np
            try:
                import numpy as np
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise BackendError(
                    "onnx_nli provider requires optional dependencies: install `probe[onnx]`"
                ) from exc

            if not self.endpoint.onnx_path or not self.endpoint.tokenizer_path:
                raise BackendError("onnx_nli requires both `onnx_path` and `tokenizer_path` in config")

            try:
                tok = Tokenizer.from_file(self.endpoint.tokenizer_path)
                tok.enable_truncation(max_length=self.endpoint.max_length, strategy="only_first")
                tok.enable_padding(length=self.endpoint.max_length)
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                sess = ort.InferenceSession(
                    self.endpoint.onnx_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise BackendError(f"Failed to load ONNX NLI artifacts: {exc}") from exc

            self._onnx_input_names = {inp.name for inp in sess.get_inputs()}
            self._onnx_ent_idx, self._onnx_con_idx = _read_onnx_label_indices(self.endpoint.onnx_path)
            self._onnx_session = sess
            self._onnx_tokenizer = tok
            self._onnx_np = np
            return sess, tok, np

    def _eval_onnx_nli(self, checks: list[str], snapshot_json: str, analysis: str) -> list[float]:
        sess, tok, np = self._ensure_onnx_loaded()
        premise = _format_onnx_premise(snapshot_json, analysis, self.domain_context)
        try:
            encs = tok.encode_batch([(premise, check) for check in checks])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._onnx_input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
            outputs = sess.run(None, feed)
        except Exception as exc:
            raise BackendError(f"ONNX NLI inference failed: {exc}") from exc

        arr = np.asarray(outputs[0], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 1:
            raise BackendError(f"Unexpected ONNX output shape: {arr.shape}")

        num_classes = int(arr.shape[1])
        probs_out: list[float] = []
        if num_classes == 1:
            for row in arr:
                logit = max(-60.0, min(60.0, float(row[0])))
                probs_out.append(_clamp_prob(1.0 / (1.0 + math.exp(-logit))))
            return probs_out

        if self.endpoint.entailment_index is not None:
            ent_idx = self.endpoint.entailment_index
        elif self._onnx_ent_idx is not None:
            ent_idx = self._onnx_ent_idx
        else:
            ent_idx = 2 if num_classes >= 3 else 1

        if self.endpoint.contradiction_index is not None:
            con_idx = self.endpoint.contradiction_index
        elif self._onnx_con_idx is not None:
            con_idx = self._onnx_con_idx
        else:
            con_idx = 0

        if not (0 <= ent_idx < num_classes and 0 <= con_idx < num_classes):
            raise BackendError(
                f"ONNX label indices ({con_idx}, {ent_idx}) out of range for {num_classes} classes"
            )
        if ent_idx == con_idx:
            raise BackendError(
                f"ONNX entailment_index and contradiction_index cannot both be {ent_idx}"
            )

        output_mode = (self.endpoint.onnx_output or "logits").strip().lower()
        for row in arr:
            if output_mode == "probabilities":
                if np.any(row < 0.0) or not math.isclose(float(np.sum(row)), 1.0, abs_tol=0.05):
                    raise BackendError(
                        "ONNX model configured with onnx_output='probabilities' returned non-probability row"
                    )
                row_probs = row
            else:
                shifted = row - np.max(row)
                exps = np.exp(shifted)
                row_probs = exps / np.sum(exps)

            p_ent = float(row_probs[ent_idx])
            if num_classes >= 3:
                score = p_ent
            else:
                p_con = float(row_probs[con_idx])
                denom = p_ent + p_con
                score = (p_ent / denom) if denom > 1e-9 else 0.5
            probs_out.append(_clamp_prob(score))
        return probs_out


def _read_onnx_label_indices(onnx_path: str) -> tuple[int | None, int | None]:
    """Read optional adjacent HuggingFace config.json label mapping for NLI heads."""
    cfg_path = Path(onnx_path).parent / "config.json"
    if not cfg_path.is_file():
        return None, None
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(raw, dict):
        return None, None

    ent_idx: int | None = None
    con_idx: int | None = None
    label2id = raw.get("label2id")
    if isinstance(label2id, dict):
        for label, idx in label2id.items():
            if not isinstance(idx, int) or isinstance(idx, bool):
                continue
            low = str(label).strip().lower()
            if low == "entailment":
                ent_idx = idx
            elif low in {"contradiction", "not_entailment"}:
                con_idx = idx
    id2label = raw.get("id2label")
    if (ent_idx is None or con_idx is None) and isinstance(id2label, dict):
        for idx_str, label in id2label.items():
            try:
                idx = int(idx_str)
            except (TypeError, ValueError):
                continue
            low = str(label).strip().lower()
            if ent_idx is None and low == "entailment":
                ent_idx = idx
            elif con_idx is None and low in {"contradiction", "not_entailment"}:
                con_idx = idx
    return ent_idx, con_idx


def _format_onnx_premise(snapshot_json: str, analysis: str, domain_context: str = "") -> str:
    """Place analysis, domain context, and observed outputs/watched vars first so truncation keeps key evidence."""
    parts: list[str] = []
    if analysis:
        parts.append(f"Analysis: {analysis}")
    ctx = domain_context.strip()
    if ctx:
        parts.append(f"Domain context: {ctx}")
    try:
        parsed = json.loads(snapshot_json)
    except ValueError:
        parsed = None

    if isinstance(parsed, dict):
        keys = set(parsed.keys())
        if "inputs" in keys and "output" in keys and keys <= {"inputs", "output", "exception"}:
            parts.append(
                f"Observed output (under test): {json.dumps({'output': parsed['output']}, ensure_ascii=False)}"
            )
            if parsed.get("exception") is not None:
                parts.append(
                    f"Observed exception: {json.dumps({'exception': parsed['exception']}, ensure_ascii=False)}"
                )
            parts.append(
                f"Call inputs (context only): {json.dumps({'inputs': parsed['inputs']}, ensure_ascii=False)}"
            )
        elif keys == {"watched"}:
            parts.append(
                f"Watched variables (under test): {json.dumps({'watched': parsed['watched']}, ensure_ascii=False)}"
            )
        else:
            ordered: dict[str, Any] = {}
            for key in ("output", "watched", "exception", "inputs"):
                if key in parsed:
                    ordered[key] = parsed[key]
            for key, val in parsed.items():
                if key not in ordered:
                    ordered[key] = val
            parts.append(json.dumps(ordered, ensure_ascii=False))
    else:
        parts.append(snapshot_json)
    return "\n".join(parts)


class TextBackend:
    """OpenAI-compatible LLM client for text-probe analysis, failure explanations, and pre-run checks."""

    def __init__(
        self,
        endpoint: ModelEndpointConfig,
        text_cfg: TextConfig | None = None,
        *,
        domain_context: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        provider = (endpoint.provider or "").strip().lower()
        if provider != "openai_compat":
            raise BackendError(
                f"Unsupported text provider {endpoint.provider!r}: expected 'openai_compat'"
            )
        self.endpoint = endpoint
        self.text_cfg = text_cfg or TextConfig()
        self.domain_context = domain_context.strip()
        self._client = httpx.Client(
            timeout=httpx.Timeout(endpoint.timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        include_user_system: bool = True,
        output_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        url = _chat_completions_url(self.endpoint.url)
        full_system = (
            compose_system_prompt(system, self.text_cfg.system_prompt) if include_user_system else system
        )
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "temperature": temperature,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else self.text_cfg.max_tokens
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        full_system = _apply_output_schema(
            payload,
            full_system,
            output_schema,
            url,
            model=self.endpoint.model,
        )
        payload["messages"] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user},
        ]
        _apply_reasoning_effort(payload, self.endpoint, url)
        try:
            data = _post_json_with_retry(
                self._client,
                url,
                payload,
                self.endpoint,
                role_label="Text LLM",
            )
        except BackendError as exc:
            if (
                output_schema is not None
                and payload.get("response_format", {}).get("type") == "json_schema"
                and "HTTP 400" in str(exc)
            ):
                schema_json = json.dumps(output_schema["schema"], ensure_ascii=False)
                fallback_system = (
                    f"{full_system}\n\nRespond with a JSON object matching this JSON Schema:\n{schema_json}"
                )
                fallback_payload = dict(payload)
                fallback_payload["response_format"] = {"type": "json_object"}
                fallback_payload["messages"] = [
                    {"role": "system", "content": fallback_system},
                    {"role": "user", "content": user},
                ]
                data = _post_json_with_retry(
                    self._client,
                    url,
                    fallback_payload,
                    self.endpoint,
                    role_label="Text LLM",
                )
            else:
                raise
        cleaned = strip_thinking(_extract_chat_content(data))
        if not cleaned:
            raise BackendError("Text model returned an empty response")
        return cleaned

    def analyze_snapshot(self, *, probe_id: str, checks: Sequence[str], snapshot_json: str) -> str:
        """Produce factual prose analysis for probe.type.text before decision scoring."""
        system, user = format_analyze_snapshot_prompts(
            probe_id=probe_id,
            checks=checks,
            snapshot_json=snapshot_json,
            detail=self.text_cfg.detail,
            domain_context=self.domain_context,
        )
        return self._chat(system, user)

    def explain_failure(
        self,
        *,
        probe_id: str,
        scores: Sequence[CheckScore],
        snapshot_json: str,
    ) -> str:
        """Write failure diagnostic citing failed check(s) and snapshot evidence."""
        failed = [s for s in scores if not s.passed] or list(scores)
        failed_lines = [f"- [FAIL p={s.probability:.2f}] {s.check}" for s in failed]
        system, user = format_explain_failure_prompts(
            probe_id=probe_id,
            failed_lines=failed_lines,
            snapshot_json=snapshot_json,
            detail=self.text_cfg.detail,
            domain_context=self.domain_context,
        )
        return self._chat(system, user)

    def analyze_checks_gate2(
        self,
        *,
        probe_id: str,
        site: str,
        checks: Sequence[str],
        signature: str,
        source_excerpt: str,
        watched_keys: Sequence[str] = (),
        ignored_params: Sequence[str] = (),
        captured_params: Sequence[str] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Gate 2 step 1: produce factual analysis of checks before decision noul scoring."""
        system, user = format_gate2_analysis_prompts(
            probe_id=probe_id,
            site=site,
            checks=checks,
            signature=signature,
            source_excerpt=source_excerpt,
            watched_keys=watched_keys,
            ignored_params=ignored_params,
            captured_params=captured_params,
            domain_context=self.domain_context,
        )
        return self._chat(
            system,
            user,
            temperature=temperature,
            include_user_system=False,
        )

    def generate_calibration_fixtures_gate3(
        self,
        *,
        probe_id: str,
        site: str = "decorator",
        checks: Sequence[str],
        signature: str,
        source_excerpt: str,
        watched_keys: Sequence[str] = (),
        ignored_params: Sequence[str] = (),
        captured_params: Sequence[str] | None = None,
        temperature: float = 0.0,
        retry_feedback: str = "",
    ) -> tuple[str, str]:
        """Gate 3 synthesis: create one valid passing snapshot and one realistic near-miss failing snapshot."""
        system, user = format_gate3_fixtures_prompts(
            probe_id=probe_id,
            site=site,
            checks=checks,
            signature=signature,
            source_excerpt=source_excerpt,
            watched_keys=watched_keys,
            ignored_params=ignored_params,
            captured_params=captured_params,
            domain_context=self.domain_context,
            retry_feedback=retry_feedback,
        )
        content = self._chat(
            system,
            user,
            include_user_system=False,
            output_schema=get_output_schema("check_gate3_fixtures"),
            temperature=temperature,
        )
        parsed = extract_json_object(content)
        passing = parsed.get("passing_snapshot")
        failing = parsed.get("failing_snapshot")
        if passing is None or failing is None:
            raise BackendError(
                f"Gate 3 synthesis missing passing_snapshot or failing_snapshot for {probe_id}"
            )
        return json.dumps(passing, ensure_ascii=False), json.dumps(failing, ensure_ascii=False)


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendError(f"Missing choices in OpenAI response: {data!r}")
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice0.get("message")
    if isinstance(msg, dict) and msg.get("refusal"):
        raise BackendError(f"Text model refused request: {msg['refusal']}")
    if choice0.get("finish_reason") == "length":
        raw_partial = (
            msg["content"]
            if isinstance(msg, dict) and isinstance(msg.get("content"), str)
            else (choice0["text"] if isinstance(choice0.get("text"), str) else "")
        )
        raise ModelTruncatedError(
            (
                "Text model response truncated (finish_reason='length'); "
                "increase [text].max_tokens or unset it to use the model default"
            ),
            partial_text=strip_thinking(raw_partial),
        )
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    text = choice0.get("text")
    if isinstance(text, str):
        return text
    raise BackendError(f"Missing message content in OpenAI response: {data!r}")


def _lookup_prob_dict(prob_map: dict[Any, Any], checks: list[str], *, kind: str) -> list[float]:
    out: list[float] = []
    for i, c in enumerate(checks):
        found = False
        raw_val: Any = None
        for candidate_key in (c, f"q{i}", f"check_{i}", str(i)):
            if candidate_key in prob_map:
                raw_val = prob_map[candidate_key]
                found = True
                break
        if not found:
            raise BackendError(f"Decision '{kind}' map missing entry for check [{i}] {c!r}")
        out.append(_clamp_prob(raw_val))
    return out


def _parse_probabilities_payload(data: dict[str, Any], checks: list[str]) -> list[float]:
    n = len(checks)
    if isinstance(data.get("probabilities"), list):
        raw_probs = data["probabilities"]
        if len(raw_probs) != n:
            raise BackendError(f"Expected {n} probabilities, got {len(raw_probs)}")
        return [_clamp_prob(p) for p in raw_probs]

    if isinstance(data.get("probabilities"), dict):
        return _lookup_prob_dict(data["probabilities"], checks, kind="probabilities")

    if isinstance(data.get("scores"), list):
        raw_scores = data["scores"]
        if len(raw_scores) != n:
            raise BackendError(f"Expected {n} scores, got {len(raw_scores)}")
        out: list[float] = []
        for item in raw_scores:
            if isinstance(item, dict):
                out.append(_clamp_prob(item.get("probability", item.get("score", item.get("p")))))
            else:
                out.append(_clamp_prob(item))
        return out

    if isinstance(data.get("results"), list):
        results = data["results"]
        if len(results) != n:
            raise BackendError(f"Expected {n} results, got {len(results)}")
        return [
            _clamp_prob(r.get("probability", r.get("score", r.get("p"))) if isinstance(r, dict) else r)
            for r in results
        ]

    if isinstance(data.get("noul"), dict):
        return _lookup_prob_dict(data["noul"], checks, kind="noul")

    if n == 1 and any(k in data for k in ("probability", "score", "p")):
        val = data.get("probability", data.get("score", data.get("p")))
        return [_clamp_prob(val)]

    raise BackendError(f"Unrecognized decision response shape: {data!r}")
