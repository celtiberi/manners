"""Prompt catalog loader and message/noul builders backed by prompts.json."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(slots=True, frozen=True)
class PromptEntry:
    """Single prompt or noul set loaded from prompts.json with role/stage metadata."""

    key: str
    role: str
    stage: str
    description: str
    prompt: str = ""
    nouls: tuple[str, ...] = ()


_DETAIL_KEY_MAP: dict[str, str] = {
    "concise": "detail_concise",
    "standard": "detail_standard",
    "detailed": "detail_detailed",
}


@cache
def load_prompt_catalog() -> dict[str, PromptEntry]:
    """Load and validate all entries from bundled probe/prompts.json."""
    raw = resources.files("probe").joinpath("prompts.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("prompts.json must contain a top-level JSON object")

    catalog: dict[str, PromptEntry] = {}
    for key, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {key!r} must be a JSON object")

        role = str(item.get("role", "")).strip()
        stage = str(item.get("stage", "")).strip()
        description = str(item.get("description", "")).strip()
        if not role or not stage or not description:
            raise ValueError(f"Prompt entry {key!r} must define role, stage, and description")

        raw_prompt = item.get("prompt")
        if isinstance(raw_prompt, list):
            prompt_text = "\n".join(str(line) for line in raw_prompt).strip()
        elif raw_prompt is not None:
            prompt_text = str(raw_prompt).strip()
        else:
            prompt_text = ""

        raw_nouls = item.get("nouls")
        if raw_nouls is None:
            nouls: tuple[str, ...] = ()
        elif isinstance(raw_nouls, list):
            nouls = tuple(str(x).strip() for x in raw_nouls if str(x).strip())
        else:
            raise ValueError(f"Prompt entry {key!r} field 'nouls' must be a list of strings")

        if role == "models.decision" and not nouls:
            raise ValueError(f"Decision entry {key!r} must define non-empty 'nouls'")
        if role == "models.text" and not prompt_text:
            raise ValueError(f"Text entry {key!r} must define non-empty 'prompt'")

        catalog[str(key)] = PromptEntry(
            key=str(key),
            role=role,
            stage=stage,
            description=description,
            prompt=prompt_text,
            nouls=nouls,
        )
    return catalog


def get_prompt(key: str) -> str:
    """Return the rendered text-model prompt string for a named entry in prompts.json."""
    entry = load_prompt_catalog()[key]
    if not entry.prompt:
        raise ValueError(f"Catalog entry {key!r} has no text prompt")
    return entry.prompt


def get_nouls(key: str) -> tuple[str, ...]:
    """Return the discrete noul proposition tuple for a named entry in prompts.json."""
    entry = load_prompt_catalog()[key]
    if not entry.nouls:
        raise ValueError(f"Catalog entry {key!r} has no decision nouls")
    return entry.nouls


def detail_instruction(detail: str) -> str:
    """Return per-check length/detail guidance for the configured text detail level."""
    mapped = _DETAIL_KEY_MAP.get(detail.strip().lower(), "detail_standard")
    return get_prompt(mapped)


def compose_system_prompt(base_system: str, user_system_prompt: str = "") -> str:
    """Prepend optional user [text].system_prompt before the task system prompt."""
    extra = user_system_prompt.strip()
    if not extra:
        return base_system
    return f"{extra}\n\n{base_system}".strip()


def _escape_json_closing_tags(snapshot_json: str) -> str:
    """Escape `</` inside JSON text so string values cannot close XML wrapper tags."""
    return snapshot_json.replace("</", "<\\/")


def _escape_xml_closing_tag(text: str, tag: str) -> str:
    """Escape closing `</tag>` occurrences in free text/code so wrappers stay balanced."""
    return re.sub(rf"</({re.escape(tag)})\s*>", r"<\/\1>", text, flags=re.IGNORECASE)


def format_typesafe_state(probe_id: str, snapshot_json: str, analysis: str = "") -> str:
    """Format the serialized state string sent to TypeSafe Jev /v1/systemone."""
    parts = [f"Probe: {probe_id}"]
    if analysis:
        safe_analysis = _escape_xml_closing_tag(analysis, "analysis")
        parts.append(f"Analysis:\n<analysis>\n{safe_analysis}\n</analysis>")
    safe_snapshot = _escape_json_closing_tags(snapshot_json)
    parts.append(f"Runtime Snapshot (JSON):\n<snapshot>\n{safe_snapshot}\n</snapshot>")
    return "\n\n".join(parts)


def format_analyze_snapshot_prompts(
    *,
    probe_id: str,
    checks: Sequence[str],
    snapshot_json: str,
    detail: str = "standard",
) -> tuple[str, str]:
    """Return (system, user) prompts for `probe.type.text` snapshot evidence extraction."""
    numbered = "\n".join(f"- [{i}] {c}" for i, c in enumerate(checks))
    system = f"{get_prompt('text_analyze_snapshot')}\n{detail_instruction(detail)}"
    safe_snapshot = _escape_json_closing_tags(snapshot_json)
    user = f"Probe: {probe_id}\nChecks:\n{numbered}\n\n<snapshot>\n{safe_snapshot}\n</snapshot>"
    return system, user


def format_explain_failure_prompts(
    *,
    probe_id: str,
    failed_lines: Sequence[str],
    snapshot_json: str,
    detail: str = "standard",
) -> tuple[str, str]:
    """Return (system, user) prompts for explaining failed decision checks."""
    joined = "\n".join(failed_lines)
    system = f"{get_prompt('text_explain_failure')}\n{detail_instruction(detail)}"
    safe_snapshot = _escape_json_closing_tags(snapshot_json)
    user = f"Probe: {probe_id}\nFailed check(s):\n{joined}\n\n<snapshot>\n{safe_snapshot}\n</snapshot>"
    return system, user


def _format_gate_user_message(
    *,
    probe_id: str,
    site: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
    watched_keys: Sequence[str] = (),
) -> str:
    numbered = "\n".join(f"- [{i}] {c}" for i, c in enumerate(checks))
    lines = [
        f"Probe: {probe_id}",
        f"Site: {site}",
        f"Signature: {signature}",
    ]
    if site == "here" and watched_keys:
        lines.append(f"Watched keys: {', '.join(watched_keys)}")
    lines.append(f"Checks:\n{numbered}")
    header = "\n".join(lines)
    safe_excerpt = _escape_xml_closing_tag(source_excerpt, "source_excerpt")
    return f"{header}\n\n<source_excerpt>\n{safe_excerpt}\n</source_excerpt>"


def format_gate2_analysis_prompts(
    *,
    probe_id: str,
    site: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
    watched_keys: Sequence[str] = (),
) -> tuple[str, str]:
    """Return (system, user) prompts for Gate 2 factual analysis before decision noul scoring."""
    return (
        get_prompt("check_gate2_analysis"),
        _format_gate_user_message(
            probe_id=probe_id,
            site=site,
            checks=checks,
            signature=signature,
            source_excerpt=source_excerpt,
            watched_keys=watched_keys,
        ),
    )


def format_gate3_fixtures_prompts(
    *,
    probe_id: str,
    site: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
    watched_keys: Sequence[str] = (),
) -> tuple[str, str]:
    """Return (system, user) prompts for Gate 3 synthetic pass/fail fixture generation."""
    return (
        get_prompt("check_gate3_fixtures"),
        _format_gate_user_message(
            probe_id=probe_id,
            site=site,
            checks=checks,
            signature=signature,
            source_excerpt=source_excerpt,
            watched_keys=watched_keys,
        ),
    )


__all__ = [
    "PromptEntry",
    "compose_system_prompt",
    "detail_instruction",
    "format_analyze_snapshot_prompts",
    "format_explain_failure_prompts",
    "format_gate2_analysis_prompts",
    "format_gate3_fixtures_prompts",
    "format_typesafe_state",
    "get_nouls",
    "get_prompt",
    "load_prompt_catalog",
]
