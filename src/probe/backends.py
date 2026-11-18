"""Decision and text model backends for runtime grading and pre-run checks."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from probe.config import DecisionConfig, ModelEndpointConfig, TextConfig

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_DETAIL_GUIDANCE: dict[str, str] = {
    "concise": "Respond concisely in 1-2 sentences citing only concrete facts from the snapshot.",
    "standard": "Respond clearly in a short paragraph citing exact fields or quotes from the snapshot.",
    "detailed": "Provide a thorough breakdown referencing each relevant field and quote from the snapshot.",
}

_DEFAULT_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_DEFAULT_TYPESAFE_MODEL = "jev-latest"
_LEGACY_JEV_MODELS = frozenset({"", "jev", "typesafe-ai/jev"})
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
_MAX_HTTP_ATTEMPTS = 2
_RETRY_BACKOFF_S = 0.4
_USER_AGENT = "probe/0.1"


class BackendError(RuntimeError):
    """Raised when a configured decision or text model backend fails or returns invalid data."""


@dataclass(slots=True, frozen=True)
class CheckScore:
    """Calibrated binary verdict for a single discrete check."""

    check: str
    probability: float
    passed: bool


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models."""
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text, ignoring <think> tags and fences."""
    cleaned = strip_thinking(text)
    candidates: list[str] = [cleaned]

    for match in _CODE_FENCE_RE.finditer(cleaned):
        candidates.insert(0, match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: locate balanced outer braces inside prose.
        start = candidate.find("{")
        while start != -1:
            depth = 0
            in_str = False
            escape = False
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
                        snippet = candidate[start : idx + 1]
                        try:
                            parsed = json.loads(snippet)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            break
            start = candidate.find("{", start + 1)

    raise BackendError(f"Expected JSON object in model response, got: {cleaned[:200]!r}")


def _clamp_prob(value: Any) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"Invalid probability value: {value!r}") from exc
    if not math.isfinite(p):
        raise BackendError(f"Non-finite probability value: {value!r}")
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


def _apply_reasoning_effort(
    payload: dict[str, Any],
    endpoint: ModelEndpointConfig,
    url: str,
) -> None:
    effort = (endpoint.reasoning_effort or "").strip()
    if not effort and "generativelanguage.googleapis.com" in url.lower():
        effort = "none"
    if effort:
        payload["reasoning_effort"] = effort


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
        except httpx.TransportError as exc:
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
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.pass_threshold = (decision_cfg or DecisionConfig()).pass_threshold
        self._client = httpx.Client(timeout=endpoint.timeout_s, transport=transport)
        self._onnx_session: Any = None
        self._onnx_tokenizer: Any = None

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
        elif provider == "logprob_bool":
            probs = self._eval_logprob_bool(probe_id, check_list, snapshot_json, analysis)
        elif provider == "openai_compat":
            probs = self._eval_openai_compat(probe_id, check_list, snapshot_json, analysis)
        elif provider == "onnx_nli":
            probs = self._eval_onnx_nli(check_list, snapshot_json, analysis)
        else:
            raise BackendError(f"Unknown decision provider: {self.endpoint.provider!r}")

        return [
            CheckScore(
                check=check,
                probability=round(p, 4),
                passed=p > self.pass_threshold,
            )
            for check, p in zip(check_list, probs, strict=True)
        ]

    def evaluate_condition(
        self,
        *,
        probe_id: str,
        condition: str,
        snapshot_json: str,
        analysis: str = "",
    ) -> bool:
        """Evaluate a `sync="when <condition>"` expression against the current snapshot."""
        cond = condition.strip()
        if cond.lower().startswith("when "):
            cond = cond[5:].strip()
        prompt_check = f"This condition is currently TRUE: {cond}"
        scores = self.evaluate_checks(
            probe_id=f"{probe_id}#sync_when",
            checks=[prompt_check],
            snapshot_json=snapshot_json,
            analysis=analysis,
        )
        return scores[0].passed if scores else False

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

        state_parts = [f"Probe: {probe_id}"]
        if analysis:
            state_parts.append(f"Analysis:\n{analysis}")
        state_parts.append(f"Runtime Snapshot (JSON):\n{snapshot_json}")
        state_str = "\n\n".join(state_parts)

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
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "probe_id": probe_id,
            "checks": checks,
            "snapshot": snapshot_json,
        }
        if analysis:
            payload["analysis"] = analysis

        data = self._post_json(self.endpoint.url, payload)
        return _parse_probabilities_payload(data, checks)

    def _eval_logprob_bool(
        self,
        probe_id: str,
        checks: list[str],
        snapshot_json: str,
        analysis: str,
    ) -> list[float]:
        url = _chat_completions_url(self.endpoint.url)
        probs: list[float] = []
        for idx, check in enumerate(checks):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a binary verifier. Reply with strictly one uppercase word: YES if the check "
                        "clearly holds for the snapshot, or NO if it does not."
                    ),
                },
                {
                    "role": "user",
                    "content": _format_eval_context(probe_id, check, snapshot_json, analysis),
                },
            ]
            data = self._post_json(
                url,
                {
                    "model": self.endpoint.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 3,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
            )
            p = _extract_yes_no_logprob(data)
            if p is None:
                # Server/model omitted logprobs; fall back cleanly to JSON probability scoring for remaining checks.
                fallback = self._eval_openai_compat(probe_id, checks[idx:], snapshot_json, analysis)
                probs.extend(fallback)
                return probs
            probs.append(p)
        return probs

    def _eval_openai_compat(
        self,
        probe_id: str,
        checks: list[str],
        snapshot_json: str,
        analysis: str,
    ) -> list[float]:
        url = _chat_completions_url(self.endpoint.url)
        numbered = "\n".join(f"{i}: {chk}" for i, chk in enumerate(checks))
        analysis_block = f"\nPrior analysis:\n{analysis}\n" if analysis else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "Evaluate each numbered check against the snapshot and return ONLY valid JSON: "
                    '{"probabilities": [<float 0.0..1.0>, ...]} with one probability per check in exact order.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Probe: {probe_id}\n"
                    f"Snapshot:\n{snapshot_json}\n"
                    f"{analysis_block}\n"
                    f"Checks:\n{numbered}"
                ),
            },
        ]
        chat_payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 300,
        }
        _apply_reasoning_effort(chat_payload, self.endpoint, url)
        data = self._post_json(url, chat_payload)
        content = _extract_chat_content(data)
        parsed = extract_json_object(content)
        return _parse_probabilities_payload(parsed, checks)

    def _ensure_onnx_loaded(self) -> tuple[Any, Any]:
        if self._onnx_session is not None and self._onnx_tokenizer is not None:
            return self._onnx_session, self._onnx_tokenizer
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise BackendError(
                "onnx_nli provider requires optional dependencies: install `probe[onnx]`"
            ) from exc

        if not self.endpoint.onnx_path or not self.endpoint.tokenizer_path:
            raise BackendError("onnx_nli requires both `onnx_path` and `tokenizer_path` in config")

        tok = Tokenizer.from_file(self.endpoint.tokenizer_path)
        tok.enable_truncation(max_length=self.endpoint.max_length)
        tok.enable_padding(length=self.endpoint.max_length)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        sess = ort.InferenceSession(
            self.endpoint.onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._onnx_session = sess
        self._onnx_tokenizer = tok
        return sess, tok

    def _eval_onnx_nli(self, checks: list[str], snapshot_json: str, analysis: str) -> list[float]:
        import numpy as np

        sess, tok = self._ensure_onnx_loaded()
        premise = f"{analysis}\nSnapshot: {snapshot_json}" if analysis else snapshot_json
        encs = tok.encode_batch([(premise, check) for check in checks])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        outputs = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        arr = np.asarray(outputs[-1], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        probs_out: list[float] = []
        ent_idx = self.endpoint.entailment_index
        con_idx = self.endpoint.contradiction_index
        for row in arr:
            # Softmax when raw logits (rows not summing near 1 or negative values present).
            if np.any(row < 0.0) or not math.isclose(float(np.sum(row)), 1.0, abs_tol=0.05):
                shifted = row - np.max(row)
                exps = np.exp(shifted)
                row_probs = exps / np.sum(exps)
            else:
                row_probs = row
            p_ent = float(row_probs[ent_idx])
            p_con = float(row_probs[con_idx])
            denom = p_ent + p_con
            score = (p_ent / denom) if denom > 1e-9 else 0.5
            probs_out.append(_clamp_prob(score))
        return probs_out


class TextBackend:
    """OpenAI-compatible LLM client for text-probe analysis, failure explanations, and pre-run checks."""

    def __init__(
        self,
        endpoint: ModelEndpointConfig,
        text_cfg: TextConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.text_cfg = text_cfg or TextConfig()
        self._client = httpx.Client(timeout=endpoint.timeout_s, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _chat(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        url = _chat_completions_url(self.endpoint.url)
        full_system = (
            f"{self.text_cfg.system_prompt.strip()}\n\n{system}".strip()
            if self.text_cfg.system_prompt.strip()
            else system
        )
        payload = {
            "model": self.endpoint.model,
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens if max_tokens is not None else self.text_cfg.max_tokens,
        }
        _apply_reasoning_effort(payload, self.endpoint, url)
        data = _post_json_with_retry(
            self._client,
            url,
            payload,
            self.endpoint,
            role_label="Text LLM",
        )
        return strip_thinking(_extract_chat_content(data))

    def _detail_instruction(self) -> str:
        return _DETAIL_GUIDANCE.get(self.text_cfg.detail.strip().lower(), _DETAIL_GUIDANCE["standard"])

    def analyze_snapshot(self, *, probe_id: str, checks: Sequence[str], snapshot_json: str) -> str:
        """Produce factual prose analysis for probe.type.text before decision scoring."""
        numbered = "\n".join(f"- {c}" for c in checks)
        system = (
            "You inspect runtime function snapshots against plain-English checks. "
            f"State whether each check is satisfied based strictly on observed data. {self._detail_instruction()}"
        )
        user = f"Probe: {probe_id}\nChecks:\n{numbered}\n\nSnapshot:\n{snapshot_json}"
        return self._chat(system, user)

    def explain_failure(
        self,
        *,
        probe_id: str,
        scores: Sequence[CheckScore],
        snapshot_json: str,
    ) -> str:
        """Write failure diagnostic citing failed check(s) and snapshot evidence."""
        score_lines = "\n".join(
            f"- [{'PASS' if s.passed else 'FAIL'} p={s.probability:.2f}] {s.check}"
            for s in scores
        )
        system = (
            "Explain why the failing check(s) contradicted the runtime snapshot. "
            f"{self._detail_instruction()}"
        )
        user = f"Probe: {probe_id}\nCheck results:\n{score_lines}\n\nSnapshot:\n{snapshot_json}"
        return self._chat(system, user)

    def audit_checks_gate2(
        self,
        *,
        probe_id: str,
        checks: Sequence[str],
        signature: str,
        source_excerpt: str,
    ) -> tuple[bool, str]:
        """Gate 2 audit: reject checks better written as Python asserts/regexes or unfalsifiable prose."""
        numbered = "\n".join(f"- {c}" for c in checks)
        system = (
            "You audit @probe checks before deployment. Reject a probe ONLY if any check is trivially "
            "a deterministic check that belongs in normal Python assert/regex/type checks (math invariants, "
            "null/None checks, list length, exact regex format) OR is vague and unfalsifiable.\n"
            'Return strictly JSON: {"approved": true|false, "reason": "<short reason>"}.'
        )
        user = (
            f"Probe: {probe_id}\n"
            f"Signature: {signature}\n"
            f"Checks:\n{numbered}\n\n"
            f"Source excerpt:\n{source_excerpt}"
        )
        content = self._chat(system, user, max_tokens=max(250, self.text_cfg.max_tokens))
        parsed = extract_json_object(content)
        approved = bool(parsed.get("approved", False))
        reason = str(parsed.get("reason", "")).strip()
        return approved, reason

    def generate_calibration_fixtures_gate3(
        self,
        *,
        probe_id: str,
        checks: Sequence[str],
        signature: str,
        source_excerpt: str,
    ) -> tuple[str, str]:
        """Gate 3 synthesis: create one valid passing snapshot and one clearly broken failing snapshot."""
        numbered = "\n".join(f"- {c}" for c in checks)
        system = (
            "Generate two synthetic JSON snapshots matching how this probe is serialized at runtime:\n"
            '- For a @probe decorator, each snapshot MUST be {"inputs": {...}, "output": ...}.\n'
            '- For a probe.here(...) watchpoint, each snapshot MUST be {"watched": {...}}.\n'
            '1) "passing_snapshot": realistic valid runtime data that clearly satisfies ALL checks.\n'
            '2) "failing_snapshot": realistic broken runtime data that clearly violates at least one check.\n'
            'Return strictly JSON: {"passing_snapshot": {...}, "failing_snapshot": {...}}.'
        )
        user = (
            f"Probe: {probe_id}\n"
            f"Signature: {signature}\n"
            f"Checks:\n{numbered}\n\n"
            f"Source excerpt:\n{source_excerpt}"
        )
        content = self._chat(system, user, max_tokens=max(600, self.text_cfg.max_tokens))
        parsed = extract_json_object(content)
        passing = parsed.get("passing_snapshot")
        failing = parsed.get("failing_snapshot")
        if passing is None or failing is None:
            raise BackendError(
                f"Gate 3 synthesis missing passing_snapshot or failing_snapshot for {probe_id}"
            )
        return json.dumps(passing, ensure_ascii=False), json.dumps(failing, ensure_ascii=False)


def _format_eval_context(probe_id: str, check: str, snapshot_json: str, analysis: str) -> str:
    parts = [f"Probe: {probe_id}", f"Snapshot:\n{snapshot_json}"]
    if analysis:
        parts.append(f"Analysis:\n{analysis}")
    parts.append(f"Check: {check}")
    return "\n\n".join(parts)


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendError(f"Missing choices in OpenAI response: {data!r}")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if isinstance(text, str):
        return text
    raise BackendError(f"Missing message content in OpenAI response: {data!r}")


def _extract_yes_no_logprob(data: dict[str, Any]) -> float | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    logprobs = choice0.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content_list = logprobs.get("content")
    if not isinstance(content_list, list) or not content_list:
        return None
    first_tok = content_list[0] if isinstance(content_list[0], dict) else {}
    top = first_tok.get("top_logprobs")
    if not isinstance(top, list) or not top:
        return None

    yes_lp: float | None = None
    no_lp: float | None = None
    total_mass = 0.0
    for item in top:
        if not isinstance(item, dict):
            continue
        tok = str(item.get("token", "")).strip().upper()
        lp = item.get("logprob")
        if lp is None:
            continue
        lp_val = float(lp)
        total_mass += math.exp(lp_val)
        if tok == "YES" and (yes_lp is None or lp_val > yes_lp):
            yes_lp = lp_val
        elif tok == "NO" and (no_lp is None or lp_val > no_lp):
            no_lp = lp_val

    if yes_lp is None and no_lp is None:
        return None
    if yes_lp is not None and no_lp is not None:
        p_y = math.exp(yes_lp)
        p_n = math.exp(no_lp)
        return _clamp_prob(p_y / (p_y + p_n))
    rem = max(1e-6, 1.0 - total_mass)
    if yes_lp is not None:
        p_y = math.exp(yes_lp)
        return _clamp_prob(p_y / (p_y + rem))
    p_n = math.exp(no_lp)  # type: ignore[arg-type]
    return _clamp_prob(rem / (p_n + rem))


def _parse_probabilities_payload(data: dict[str, Any], checks: list[str]) -> list[float]:
    n = len(checks)
    if isinstance(data.get("probabilities"), list):
        raw_probs = data["probabilities"]
        if len(raw_probs) != n:
            raise BackendError(f"Expected {n} probabilities, got {len(raw_probs)}")
        return [_clamp_prob(p) for p in raw_probs]

    if isinstance(data.get("probabilities"), dict):
        prob_map = data["probabilities"]
        return [_clamp_prob(prob_map[c]) for c in checks]

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
        noul = data["noul"]
        if all(c in noul for c in checks):
            return [_clamp_prob(noul[c]) for c in checks]
        if all(f"check_{i}" in noul for i in range(n)):
            return [_clamp_prob(noul[f"check_{i}"]) for i in range(n)]

    if n == 1 and any(k in data for k in ("probability", "score", "p")):
        val = data.get("probability", data.get("score", data.get("p")))
        return [_clamp_prob(val)]

    raise BackendError(f"Unrecognized decision response shape: {data!r}")
