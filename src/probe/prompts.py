"""Prompt catalog loader and model message builders backed by prompts.json."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(slots=True, frozen=True)
class PromptEntry:
    """Single prompt entry loaded from prompts.json with role/stage metadata."""

    key: str
    role: str
    provider: str
    stage: str
    description: str
    prompt: str


_DETAIL_KEY_MAP: dict[str, str] = {
    "concise": "detail_concise",
    "standard": "detail_standard",
    "detailed": "detail_detailed",
}


@cache
def load_prompt_catalog() -> dict[str, PromptEntry]:
    """Load and validate all prompt entries from bundled probe/prompts.json."""
    raw = resources.files("probe").joinpath("prompts.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("prompts.json must contain a top-level JSON object")

    catalog: dict[str, PromptEntry] = {}
    for key, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {key!r} must be a JSON object")
        raw_prompt = item.get("prompt")
        if isinstance(raw_prompt, list):
            prompt_text = "\n".join(str(line) for line in raw_prompt).strip()
        else:
            prompt_text = str(raw_prompt or "").strip()
        catalog[str(key)] = PromptEntry(
            key=str(key),
            role=str(item.get("role", "")).strip(),
            provider=str(item.get("provider", "")).strip(),
            stage=str(item.get("stage", "")).strip(),
            description=str(item.get("description", "")).strip(),
            prompt=prompt_text,
        )
    return catalog


def get_prompt(key: str) -> str:
    """Return the rendered prompt string for a named entry in prompts.json."""
    return load_prompt_catalog()[key].prompt


def detail_instruction(detail: str) -> str:
    """Return length/detail guidance for the configured text detail level."""
    mapped = _DETAIL_KEY_MAP.get(detail.strip().lower(), "detail_standard")
    return get_prompt(mapped)


def compose_system_prompt(base_system: str, user_system_prompt: str = "") -> str:
    """Prepend optional user [text].system_prompt before the task system prompt."""
    extra = user_system_prompt.strip()
    if not extra:
        return base_system
    return f"{extra}\n\n{base_system}".strip()


def format_when_check(condition: str) -> str:
    """Convert a `when="..."` condition into a discrete decision check statement."""
    cond = condition.strip()
    if cond.lower().startswith("when "):
        cond = cond[5:].strip()
    return f"{get_prompt('when_condition')} {cond}"


def format_typesafe_state(probe_id: str, snapshot_json: str, analysis: str = "") -> str:
    """Format the serialized state string sent to TypeSafe Jev /v1/systemone."""
    parts = [f"Probe: {probe_id}"]
    if analysis:
        parts.append(f"Analysis:\n{analysis}")
    parts.append(f"Runtime Snapshot (JSON):\n{snapshot_json}")
    return "\n\n".join(parts)


def format_logprob_eval_user(probe_id: str, check: str, snapshot_json: str, analysis: str = "") -> str:
    """Format the user message for single-check YES/NO logprob verification."""
    parts = [f"Probe: {probe_id}", f"Snapshot:\n{snapshot_json}"]
    if analysis:
        parts.append(f"Analysis:\n{analysis}")
    parts.append(f"Check: {check}")
    return "\n\n".join(parts)


def format_openai_decision_user(
    probe_id: str,
    checks: Sequence[str],
    snapshot_json: str,
    analysis: str = "",
) -> str:
    """Format the user message for OpenAI-compatible multi-check JSON probability scoring."""
    numbered = "\n".join(f"{i}: {chk}" for i, chk in enumerate(checks))
    analysis_block = f"\nPrior analysis:\n{analysis}\n" if analysis else ""
    return (
        f"Probe: {probe_id}\n"
        f"Snapshot:\n{snapshot_json}\n"
        f"{analysis_block}\n"
        f"Checks:\n{numbered}"
    )


def format_analyze_snapshot_prompts(
    *,
    probe_id: str,
    checks: Sequence[str],
    snapshot_json: str,
    detail: str = "standard",
) -> tuple[str, str]:
    """Return (system, user) prompts for `probe.type.text` snapshot analysis."""
    bulleted = "\n".join(f"- {c}" for c in checks)
    system = f"{get_prompt('text_analyze_snapshot')} {detail_instruction(detail)}"
    user = f"Probe: {probe_id}\nChecks:\n{bulleted}\n\nSnapshot:\n{snapshot_json}"
    return system, user


def format_explain_failure_prompts(
    *,
    probe_id: str,
    score_lines: Sequence[str],
    snapshot_json: str,
    detail: str = "standard",
) -> tuple[str, str]:
    """Return (system, user) prompts for explaining a failed decision probe."""
    joined_scores = "\n".join(score_lines)
    system = f"{get_prompt('text_explain_failure')} {detail_instruction(detail)}"
    user = f"Probe: {probe_id}\nCheck results:\n{joined_scores}\n\nSnapshot:\n{snapshot_json}"
    return system, user


def _format_gate_prompts(
    prompt_key: str,
    *,
    probe_id: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
) -> tuple[str, str]:
    bulleted = "\n".join(f"- {c}" for c in checks)
    user = (
        f"Probe: {probe_id}\n"
        f"Signature: {signature}\n"
        f"Checks:\n{bulleted}\n\n"
        f"Source excerpt:\n{source_excerpt}"
    )
    return get_prompt(prompt_key), user


def format_gate2_audit_prompts(
    *,
    probe_id: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
) -> tuple[str, str]:
    """Return (system, user) prompts for `probe check` Gate 2 assert-vs-prose auditing."""
    return _format_gate_prompts(
        "check_gate2_audit",
        probe_id=probe_id,
        checks=checks,
        signature=signature,
        source_excerpt=source_excerpt,
    )


def format_gate3_fixtures_prompts(
    *,
    probe_id: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
) -> tuple[str, str]:
    """Return (system, user) prompts for `probe check` Gate 3 fixture synthesis."""
    return _format_gate_prompts(
        "check_gate3_fixtures",
        probe_id=probe_id,
        checks=checks,
        signature=signature,
        source_excerpt=source_excerpt,
    )


__all__ = [
    "PromptEntry",
    "compose_system_prompt",
    "detail_instruction",
    "format_analyze_snapshot_prompts",
    "format_explain_failure_prompts",
    "format_gate2_audit_prompts",
    "format_gate3_fixtures_prompts",
    "format_logprob_eval_user",
    "format_openai_decision_user",
    "format_typesafe_state",
    "format_when_check",
    "get_prompt",
    "load_prompt_catalog",
]
