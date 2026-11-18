"""Prompt catalog loader and message/noul builders backed by prompts.json."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any


@dataclass(slots=True, frozen=True)
class PromptEntry:
    """Single prompt or noul set loaded from prompts.json with role/stage metadata."""

    key: str
    role: str
    stage: str
    description: str
    prompt: str = ""
    nouls: tuple[str, ...] = ()
    expected: bool = True
    output_schema: dict[str, Any] | None = None


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

        raw_expected = item.get("expected", True)
        if not isinstance(raw_expected, bool):
            raise ValueError(f"Prompt entry {key!r} field 'expected' must be a boolean")

        raw_schema = item.get("output_schema")
        output_schema: dict[str, Any] | None = None
        if raw_schema is not None:
            if (
                not isinstance(raw_schema, dict)
                or not isinstance(raw_schema.get("name"), str)
                or not raw_schema["name"].strip()
                or not isinstance(raw_schema.get("schema"), dict)
            ):
                raise ValueError(
                    f"Prompt entry {key!r} field 'output_schema' must define non-empty 'name' and dict 'schema'"
                )
            output_schema = copy.deepcopy(raw_schema)

        catalog[str(key)] = PromptEntry(
            key=str(key),
            role=role,
            stage=stage,
            description=description,
            prompt=prompt_text,
            nouls=nouls,
            expected=raw_expected,
            output_schema=output_schema,
        )
    return catalog


def get_prompt(key: str) -> str:
    """Return the rendered text-model prompt string for a named entry in prompts.json."""
    entry = load_prompt_catalog()[key]
    if not entry.prompt:
        raise ValueError(f"Catalog entry {key!r} has no text prompt")
    return entry.prompt


def get_output_schema(key: str) -> dict[str, Any] | None:
    """Return the declarative JSON schema specification for a named entry, if any."""
    entry = load_prompt_catalog()[key]
    return copy.deepcopy(entry.output_schema) if entry.output_schema is not None else None


def get_nouls(key: str) -> tuple[str, ...]:
    """Return the discrete noul proposition tuple for a named entry in prompts.json."""
    entry = load_prompt_catalog()[key]
    if not entry.nouls:
        raise ValueError(f"Catalog entry {key!r} has no decision nouls")
    return entry.nouls


def get_noul_expected(key: str) -> bool:
    """Return the expected boolean polarity for a decision noul entry (True = needs >0.50, False = needs <0.50)."""
    entry = load_prompt_catalog()[key]
    if not entry.nouls:
        raise ValueError(f"Catalog entry {key!r} has no decision nouls")
    return entry.expected


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


def _escape_closing_tags(text: str) -> str:
    """Escape any `</` sequences inside untrusted blocks so wrapper XML tags cannot be closed early."""
    return text.replace("</", "<\\/")


def _format_snapshot_body(snapshot_json: str) -> str:
    """Foreground observed output or watched variables ahead of call inputs when structured."""
    try:
        parsed = json.loads(snapshot_json)
    except ValueError:
        return _escape_closing_tags(snapshot_json)

    if isinstance(parsed, dict):
        keys = set(parsed.keys())
        if "inputs" in keys and "output" in keys and keys <= {"inputs", "output", "exception"}:
            sections = [
                f"Observed Output (under test):\n{json.dumps(parsed['output'], ensure_ascii=False)}"
            ]
            if parsed.get("exception") is not None:
                sections.append(
                    f"Observed Exception:\n{json.dumps(parsed['exception'], ensure_ascii=False)}"
                )
            sections.append(
                f"Call Inputs (context only):\n{json.dumps(parsed['inputs'], ensure_ascii=False)}"
            )
            return _escape_closing_tags("\n\n".join(sections))
        if keys == {"watched"}:
            body = f"Watched Variables (under test):\n{json.dumps(parsed['watched'], ensure_ascii=False)}"
            return _escape_closing_tags(body)

    return _escape_closing_tags(snapshot_json)


def format_typesafe_state(
    probe_id: str,
    snapshot_json: str,
    analysis: str = "",
    domain_context: str = "",
) -> str:
    """Format the serialized state string sent to TypeSafe Jev /v1/systemone."""
    parts = [f"Probe: {probe_id}"]
    ctx = domain_context.strip()
    if ctx:
        parts.append(f"Domain Context:\n<domain_context>\n{_escape_closing_tags(ctx)}\n</domain_context>")
    if analysis:
        parts.append(f"Analysis:\n<analysis>\n{_escape_closing_tags(analysis)}\n</analysis>")
    parts.append(f"Runtime Snapshot (JSON):\n<snapshot>\n{_format_snapshot_body(snapshot_json)}\n</snapshot>")
    return "\n\n".join(parts)


def format_analyze_snapshot_prompts(
    *,
    probe_id: str,
    checks: Sequence[str],
    snapshot_json: str,
    detail: str = "standard",
    domain_context: str = "",
) -> tuple[str, str]:
    """Return (system, user) prompts for `probe.type.text` snapshot evidence extraction."""
    numbered = "\n".join(f"- [{i}] {c}" for i, c in enumerate(checks))
    system = f"{get_prompt('text_analyze_snapshot')}\n{detail_instruction(detail)}"
    parts = [f"Probe: {probe_id}", f"Checks:\n{numbered}"]
    ctx = domain_context.strip()
    if ctx:
        parts.append(f"<domain_context>\n{_escape_closing_tags(ctx)}\n</domain_context>")
    parts.append(f"<snapshot>\n{_format_snapshot_body(snapshot_json)}\n</snapshot>")
    return system, "\n\n".join(parts)


def format_explain_failure_prompts(
    *,
    probe_id: str,
    failed_lines: Sequence[str],
    snapshot_json: str,
    detail: str = "standard",
    domain_context: str = "",
) -> tuple[str, str]:
    """Return (system, user) prompts for explaining failed decision checks."""
    joined = "\n".join(failed_lines)
    system = f"{get_prompt('text_explain_failure')}\n{detail_instruction(detail)}"
    parts = [f"Probe: {probe_id}", f"Failed check(s):\n{joined}"]
    ctx = domain_context.strip()
    if ctx:
        parts.append(f"<domain_context>\n{_escape_closing_tags(ctx)}\n</domain_context>")
    parts.append(f"<snapshot>\n{_format_snapshot_body(snapshot_json)}\n</snapshot>")
    return system, "\n\n".join(parts)


def _format_gate_user_message(
    *,
    probe_id: str,
    site: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
    watched_keys: Sequence[str] = (),
    domain_context: str = "",
    retry_feedback: str = "",
) -> str:
    numbered = "\n".join(f"- [{i}] {c}" for i, c in enumerate(checks))
    lines = [
        f"Probe: {probe_id}",
        f"Site: {site}",
    ]
    if site == "here" and watched_keys:
        lines.append(f"Watched keys: {', '.join(watched_keys)}")
    lines.append(f"Checks:\n{numbered}")
    header = "\n".join(lines)
    safe_sig = _escape_closing_tags(signature)
    safe_excerpt = _escape_closing_tags(source_excerpt)
    parts = [header]
    ctx = domain_context.strip()
    if ctx:
        parts.append(f"<domain_context>\n{_escape_closing_tags(ctx)}\n</domain_context>")
    fb = retry_feedback.strip()
    if fb:
        parts.append(f"<previous_attempt_error>\n{_escape_closing_tags(fb)}\n</previous_attempt_error>")
    parts.append(f"<source_excerpt>\nSignature: {safe_sig}\n{safe_excerpt}\n</source_excerpt>")
    return "\n\n".join(parts)


def format_gate2_analysis_prompts(
    *,
    probe_id: str,
    site: str,
    checks: Sequence[str],
    signature: str,
    source_excerpt: str,
    watched_keys: Sequence[str] = (),
    domain_context: str = "",
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
            domain_context=domain_context,
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
    domain_context: str = "",
    retry_feedback: str = "",
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
            domain_context=domain_context,
            retry_feedback=retry_feedback,
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
    "get_noul_expected",
    "get_nouls",
    "get_output_schema",
    "get_prompt",
    "load_prompt_catalog",
]
