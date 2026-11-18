"""Typer CLI for `probe`: check, list, report, log, runs, clean."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from probe.backends import BackendError
from probe.check import (
    _load_lockfile,
    _matches_file_or_dir_prefixes,
    _rel_display_path,
    _scan_files,
    compute_effective_combined_hash,
    run_probe_check,
)
from probe.config import load_config
from probe.store import EventRecord, ProbeStore, RunRecord

app = typer.Typer(
    name="probe",
    help="AI runtime checks, pre-run check, and call-ordered behavior logs.",
    no_args_is_help=True,
    add_completion=False,
)


def _open_existing_store(runs_dir: Path, *, read_only: bool = True) -> ProbeStore:
    if not runs_dir.is_dir():
        typer.echo(f"No probe runs found at {runs_dir}. Run with PROBES=1 first.", err=True)
        raise typer.Exit(code=1)
    return ProbeStore(runs_dir, create=not read_only)


def _run_status_label(run: RunRecord, events: list[EventRecord]) -> str:
    status = run.status.upper()
    if run.status == "halted" and run.halt_call_seq is not None:
        trig = next((e for e in events if e.call_seq == run.halt_call_seq), None)
        if trig is not None and trig.error_code:
            return f"HALTED on {trig.error_code} at call #{run.halt_call_seq:03d}"
        if trig is not None:
            return f"HALTED on {trig.verdict.upper()} at call #{run.halt_call_seq:03d}"
        return f"HALTED at call #{run.halt_call_seq:03d}"
    return status


def _event_to_dict(ev: EventRecord, *, include_payload: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "call_seq": ev.call_seq,
        "elapsed_ms": ev.elapsed_ms,
        "started_at": ev.started_at,
        "probe_id": ev.probe_id,
        "site": ev.site,
        "probe_type": ev.probe_type,
        "file_line": ev.file_line,
        "verdict": ev.verdict,
        "error_code": ev.error_code,
        "check_scores": [dataclasses.asdict(s) for s in ev.check_scores],
        "text_analysis": ev.text_analysis,
        "stack_excerpt": ev.stack_excerpt,
        "judged_at": ev.judged_at,
        "latency_ms": ev.latency_ms,
    }
    if include_payload:
        try:
            data["payload"] = json.loads(ev.payload_json)
        except Exception:
            data["payload"] = ev.payload_json
    return data


def _format_event_text(
    ev: EventRecord,
    *,
    halt_call_seq: int | None,
    detailed_pass: bool = False,
    include_payload: bool = False,
) -> list[str]:
    sec = ev.elapsed_ms / 1000.0
    marker = "  <-- HALT TRIGGER" if halt_call_seq == ev.call_seq else ""
    lines = [
        f"  #{ev.call_seq:03d}  +{sec:.3f}s  [{ev.verdict.upper()}]  {ev.probe_id} "
        f"({ev.file_line}) [kind={ev.probe_type}]{marker}"
    ]
    if ev.error_code:
        lines.append(f"        error: {ev.error_code}")

    if ev.check_scores:
        show_all_scores = (
            detailed_pass
            or ev.verdict != "pass"
            or ev.probe_type == "text"
            or len(ev.check_scores) <= 1
        )
        if show_all_scores:
            lines.append("        checks:")
            for s in ev.check_scores:
                mark = "✓" if s.passed else "✗"
                lines.append(f"          {mark} [{s.probability:.2f}] {s.check}")
        else:
            passed_n = sum(1 for s in ev.check_scores if s.passed)
            min_p = min(s.probability for s in ev.check_scores)
            lines.append(f"        checks:  {passed_n}/{len(ev.check_scores)} passed (min p={min_p:.2f})")

    if ev.text_analysis:
        lines.append(f"        analysis: {ev.text_analysis}")
    if ev.stack_excerpt and ev.verdict in {"fail", "error"}:
        lines.append("        stack:")
        for sline in ev.stack_excerpt.splitlines():
            lines.append(f"          {sline}")
    if include_payload:
        lines.append(f"        payload: {ev.payload_json}")
    return lines


def _format_report_text(run: RunRecord, events: list[EventRecord], *, show_all: bool) -> str:
    total = run.pass_count + run.fail_count + run.error_count
    status_label = _run_status_label(run, events)
    out = [
        f"Run {run.started_at} (id: {run.id}, cmd: {run.cmd}) — {status_label}",
        (
            f"Probes: {run.active_probes} active | Calls: {total} "
            f"({run.pass_count} pass, {run.fail_count} fail, {run.error_count} error)"
        ),
        "",
    ]

    visible = events if show_all else [e for e in events if e.verdict != "pass"]
    if not events:
        out.append("No probe calls recorded in this run.")
        return "\n".join(out)

    if not visible:
        out.append(f"All {total} probe call(s) passed. Run with `--all` to inspect every entry.")
        return "\n".join(out)

    out.append("Chronological Timeline (ordered by call_seq):")
    for idx, ev in enumerate(visible):
        if idx > 0:
            out.append("")
        out.extend(
            _format_event_text(
                ev,
                halt_call_seq=run.halt_call_seq,
                detailed_pass=False,
                include_payload=False,
            )
        )
    return "\n".join(out)


def _format_report_md(run: RunRecord, events: list[EventRecord], *, show_all: bool) -> str:
    total = run.pass_count + run.fail_count + run.error_count
    status_label = _run_status_label(run, events)
    out = [
        f"## Run `{run.id}` ({run.started_at}) — **{status_label}**",
        "",
        f"- **Command:** `{run.cmd}`",
        f"- **Probes:** {run.active_probes} active",
        f"- **Calls:** {total} ({run.pass_count} pass, {run.fail_count} fail, {run.error_count} error)",
        "",
    ]
    visible = events if show_all else [e for e in events if e.verdict != "pass"]
    if not visible:
        out.append("All probe calls passed." if events else "No probe calls recorded.")
        return "\n".join(out)

    out.append("### Chronological Timeline (`call_seq`)")
    out.append("")
    for ev in visible:
        sec = ev.elapsed_ms / 1000.0
        halt_tag = " **(HALT TRIGGER)**" if run.halt_call_seq == ev.call_seq else ""
        out.append(
            f"- `#{ev.call_seq:03d}` `+{sec:.3f}s` **`[{ev.verdict.upper()}]`** "
            f"`{ev.probe_id}` (`{ev.file_line}`, `kind={ev.probe_type}`){halt_tag}"
        )
        if ev.error_code:
            out.append(f"  - error: `{ev.error_code}`")
        for s in ev.check_scores:
            mark = "PASS" if s.passed else "FAIL"
            out.append(f"  - [{mark} p={s.probability:.2f}] {s.check}")
        if ev.text_analysis:
            out.append(f"  - analysis: {ev.text_analysis}")
        if ev.stack_excerpt and ev.verdict in {"fail", "error"}:
            out.append("  - stack:")
            out.append("    ```text")
            for sline in ev.stack_excerpt.splitlines():
                out.append(f"    {sline}")
            out.append("    ```")
    return "\n".join(out)


def _resolve_cli_scan_paths(
    paths: list[Path] | None,
    *,
    lock_path: Path | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    if not paths:
        return []
    cwd = Path.cwd()
    base_root = (project_root or cwd).resolve()
    locked: dict[str, Any] | None = None
    resolved_paths: list[Path] = []
    for p in paths:
        resolved = p.resolve() if p.is_absolute() else (cwd / p).resolve()
        if not resolved.exists():
            allow_deleted = False
            if lock_path is not None and lock_path.exists():
                if locked is None:
                    locked = _load_lockfile(lock_path).get("probes", {})
                rel = _rel_display_path(resolved, base_root)
                allow_deleted = any(
                    isinstance(meta, dict)
                    and _matches_file_or_dir_prefixes(str(meta.get("file_line", "")), {rel})
                    for meta in locked.values()
                )
            if not allow_deleted:
                typer.echo(f"Path does not exist: {p}", err=True)
                raise typer.Exit(code=1)
        resolved_paths.append(resolved)
    return resolved_paths


@app.command("check")
def check_cmd(
    paths: list[Path] = typer.Argument(None, help="Files or directories to scan (defaults to project root)."),
    force: bool = typer.Option(False, "--force", help="Re-run Gates 2-3 even when unchanged in lockfile."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
) -> None:
    """Run Gates 1-3 on new or changed probes and update .probes/lock.json."""
    try:
        cfg = load_config()
        resolved_paths = _resolve_cli_scan_paths(
            paths,
            lock_path=cfg.resolve_lock_path(),
            project_root=cfg.project_root,
        )
        ok, results = run_probe_check(resolved_paths, force=force, config=cfg)
    except (BackendError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(
            json.dumps(
                {
                    "ok": ok,
                    "count": len(results),
                    "results": [r.to_dict() for r in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        if not results:
            typer.echo("No @probe or probe.here calls found.")
        for res in results:
            tag = res.status.upper()
            gate_note = f" ({res.gate})" if res.status == "failed" else ""
            typer.echo(f"[{tag}{gate_note}] {res.probe_id} ({res.file_line}) — {res.reason}")

    if not ok:
        raise typer.Exit(code=1)


@app.command("list")
def list_cmd(
    paths: list[Path] = typer.Argument(None, help="Files or directories to scan (defaults to project root)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List statically discovered @probe and probe.here call sites and lock status."""
    try:
        cfg = load_config()
        resolved_paths = _resolve_cli_scan_paths(paths, project_root=cfg.project_root)
        _, discovered, parse_errors = _scan_files(resolved_paths, root=cfg.project_root)
        lock_path = cfg.resolve_lock_path()
        lock_data = _load_lockfile(lock_path)
    except (BackendError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if parse_errors:
        base_root = cfg.project_root.resolve()
        for pf, err in sorted(parse_errors.items()):
            typer.echo(f"Error parsing {_rel_display_path(pf, base_root)}: {err}", err=True)
        raise typer.Exit(code=1)

    locked = lock_data.get("probes", {})
    fixtures_dir = lock_path.parent / "fixtures"
    id_counts = Counter(item.probe_id for item in discovered)

    rows: list[dict[str, Any]] = []
    for item in discovered:
        effective_hash, _, fixture_error = compute_effective_combined_hash(
            item,
            fixtures_dir=fixtures_dir,
            pass_threshold=cfg.decision.pass_threshold,
        )
        entry = locked.get(item.probe_id)
        watched_ok = item.site != "here" or (
            isinstance(entry, dict) and entry.get("watched_keys") == list(item.watched_keys)
        )
        params_ok = item.site != "decorator" or (
            isinstance(entry, dict) and entry.get("param_names") == list(item.param_names)
        )
        if item.literal_error or id_counts[item.probe_id] > 1 or fixture_error is not None:
            lock_status = "invalid"
        elif not isinstance(entry, dict):
            lock_status = "unlocked"
        elif (
            entry.get("combined_hash") == effective_hash
            and entry.get("site") == item.site
            and entry.get("kind") == item.probe_type.value
            and watched_ok
            and params_ok
        ):
            lock_status = "locked"
        elif entry.get("checks_hash") == item.checks_hash:
            lock_status = "source_changed"
        else:
            lock_status = "checks_changed"

        rows.append(
            {
                "probe_id": item.probe_id,
                "file_line": item.file_line,
                "site": item.site,
                "kind": item.probe_type.value,
                "lock_status": lock_status,
                "checks": list(item.checks),
            }
        )

    if json_out:
        typer.echo(json.dumps({"probes": rows}, indent=2, ensure_ascii=False))
        return

    if not rows:
        typer.echo("No @probe or probe.here calls found.")
        return

    for row in rows:
        typer.echo(
            f"{row['probe_id']}  [{row['lock_status']}]  "
            f"({row['file_line']}, site={row['site']}, kind={row['kind']})"
        )
        for chk in row["checks"]:
            typer.echo(f"  - {chk}")


@app.command("report")
def report_cmd(
    run_id: str | None = typer.Option(None, "--run", help="Specific run ULID (defaults to latest run)."),
    show_all: bool = typer.Option(False, "--all", help="Include passed probe calls in timeline."),
    fmt: str = typer.Option("text", "--format", help="Output format: text | md | json."),
) -> None:
    """Print call-ordered summary of the latest or specified probe run."""
    cfg = load_config()
    runs_dir = cfg.resolve_runs_dir()
    store = _open_existing_store(runs_dir)
    try:
        run = store.get_run(run_id)
        if run is None:
            typer.echo(f"No matching run found in {runs_dir}.", err=True)
            raise typer.Exit(code=1)
        events = store.list_events(run.id)
    finally:
        store.close()

    fmt_norm = fmt.strip().lower()
    if fmt_norm == "json":
        visible = events if show_all else [e for e in events if e.verdict != "pass"]
        payload = {
            "run": dataclasses.asdict(run),
            "events": [_event_to_dict(e, include_payload=True) for e in visible],
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    elif fmt_norm == "md":
        typer.echo(_format_report_md(run, events, show_all=show_all))
    elif fmt_norm == "text":
        typer.echo(_format_report_text(run, events, show_all=show_all))
    else:
        typer.echo(f"Unsupported format: {fmt!r} (expected text, md, or json)", err=True)
        raise typer.Exit(code=1)


@app.command("log")
def log_cmd(
    probe_ids: list[str] = typer.Argument(..., help="One or more probe IDs or glob patterns."),
    run_id: str | None = typer.Option(None, "--run", help="Specific run ULID (defaults to latest run)."),
    payload: bool = typer.Option(False, "--payload", help="Include redacted input/output snapshots."),
    fmt: str = typer.Option("text", "--format", help="Output format: text | md | json."),
) -> None:
    """Interleave detailed chronological entries (call_seq ASC) for one or more probe IDs/globs."""
    if not probe_ids or any(not pat.strip() for pat in probe_ids):
        typer.echo("Probe ID or glob pattern must not be empty.", err=True)
        raise typer.Exit(code=1)
    cfg = load_config()
    runs_dir = cfg.resolve_runs_dir()
    store = _open_existing_store(runs_dir)
    try:
        run = store.get_run(run_id)
        if run is None:
            typer.echo(f"No matching run found in {runs_dir}.", err=True)
            raise typer.Exit(code=1)
        events = store.list_events(run.id, probe_patterns=probe_ids)
    finally:
        store.close()

    fmt_norm = fmt.strip().lower()
    if fmt_norm == "json":
        out = {
            "run": dataclasses.asdict(run),
            "patterns": probe_ids,
            "events": [_event_to_dict(e, include_payload=payload) for e in events],
        }
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if fmt_norm == "md":
        lines = [
            f"## Probe Log `{run.id}` (`{', '.join(probe_ids)}`)",
            "",
        ]
        if not events:
            lines.append("No matching events.")
        for ev in events:
            sec = ev.elapsed_ms / 1000.0
            lines.append(
                f"- `#{ev.call_seq:03d}` `+{sec:.3f}s` **`[{ev.verdict.upper()}]`** "
                f"`{ev.probe_id}` (`{ev.file_line}`)"
            )
            if ev.error_code:
                lines.append(f"  - error: `{ev.error_code}`")
            for s in ev.check_scores:
                mark = "PASS" if s.passed else "FAIL"
                lines.append(f"  - [{mark} p={s.probability:.2f}] {s.check}")
            if ev.text_analysis:
                lines.append(f"  - analysis: {ev.text_analysis}")
            if ev.stack_excerpt and ev.verdict in {"fail", "error"}:
                lines.append("  - stack:")
                lines.append("    ```text")
                for sline in ev.stack_excerpt.splitlines():
                    lines.append(f"    {sline}")
                lines.append("    ```")
            if payload:
                lines.append(f"  - payload: `{ev.payload_json}`")
        typer.echo("\n".join(lines))
        return

    if fmt_norm != "text":
        typer.echo(f"Unsupported format: {fmt!r} (expected text, md, or json)", err=True)
        raise typer.Exit(code=1)

    lines = [
        f"Run {run.started_at} (id: {run.id}) — log for {', '.join(probe_ids)} ({len(events)} calls)",
        "",
    ]
    if not events:
        lines.append("No matching probe events recorded.")
    for idx, ev in enumerate(events):
        if idx > 0:
            lines.append("")
        lines.extend(
            _format_event_text(
                ev,
                halt_call_seq=run.halt_call_seq,
                detailed_pass=True,
                include_payload=payload,
            )
        )
    typer.echo("\n".join(lines))


@app.command("runs")
def runs_cmd(
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum number of recent runs to show."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List recent probe runs with timestamps, verdict counts, and status."""
    cfg = load_config()
    store = _open_existing_store(cfg.resolve_runs_dir())
    try:
        runs = store.list_runs(limit=limit)
    finally:
        store.close()

    if json_out:
        typer.echo(json.dumps({"runs": [dataclasses.asdict(r) for r in runs]}, indent=2, ensure_ascii=False))
        return

    if not runs:
        typer.echo("No runs recorded.")
        return

    for r in runs:
        total = r.pass_count + r.fail_count + r.error_count
        typer.echo(
            f"{r.id}  {r.started_at}  [{r.status.upper()}]  "
            f"{total} calls ({r.pass_count}p/{r.fail_count}f/{r.error_count}e)  {r.cmd}"
        )


@app.command("clean")
def clean_cmd(
    keep: int = typer.Option(20, "--keep", min=0, help="Number of newest runs to retain."),
) -> None:
    """Prune old run logs from .probes/runs."""
    cfg = load_config()
    store = _open_existing_store(cfg.resolve_runs_dir(), read_only=False)
    try:
        deleted = store.clean_runs(keep=keep)
    finally:
        store.close()
    typer.echo(f"Removed {deleted} old run(s) (kept newest {keep}).")
