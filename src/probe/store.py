"""Thread-safe per-run JSONL storage for probe runs and call-ordered events."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from probe.backends import CheckScore
from probe.config import utc_now_iso


@dataclass(slots=True, frozen=True)
class RunRecord:
    """Summary metadata and verdict counts for a single probed process run."""

    id: str
    started_at: str
    finished_at: str | None
    cmd: str
    pid: int
    status: str
    halt_call_seq: int | None
    decision_model: str
    text_model: str
    pass_count: int = 0
    fail_count: int = 0
    error_count: int = 0
    active_probes: int = 0


@dataclass(slots=True, frozen=True)
class EventRecord:
    """Single evaluated probe call ordered by call_seq."""

    run_id: str
    call_seq: int
    elapsed_ms: float
    started_at: str
    probe_id: str
    site: str
    probe_type: str
    file_line: str
    checks: tuple[str, ...]
    payload_json: str
    check_scores: tuple[CheckScore, ...]
    verdict: str
    error_code: str | None
    text_analysis: str | None
    stack_excerpt: str | None
    judged_at: str
    latency_ms: float


def _ensure_probes_gitignore(runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    parent = runs_dir.parent
    if parent.name == ".probes":
        gi = parent / ".gitignore"
        if not gi.exists():
            gi.write_text("runs/\n", encoding="utf-8")


def _parse_event_record(rec: dict[str, Any], default_run_id: str) -> EventRecord | None:
    try:
        call_seq = int(rec["call_seq"])
        elapsed_ms = float(rec["elapsed_ms"])
        started_at = str(rec["started_at"])
        probe_id = str(rec["probe_id"])
        site = str(rec["site"])
        probe_type = str(rec["probe_type"])
        file_line = str(rec["file_line"])
        verdict = str(rec["verdict"])
        judged_at = str(rec.get("judged_at") or started_at)
        latency_ms = float(rec.get("latency_ms") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None

    raw_checks = rec.get("checks")
    checks = tuple(str(c) for c in raw_checks) if isinstance(raw_checks, list) else ()

    if rec.get("payload_is_raw"):
        payload_json = str(rec.get("payload", ""))
    else:
        payload_json = json.dumps(rec.get("payload"), ensure_ascii=False)

    raw_scores = rec.get("check_scores")
    scores = (
        tuple(
            CheckScore(
                check=str(item.get("check", "")),
                probability=float(item.get("probability", 0.0)),
                passed=bool(item.get("passed", False)),
            )
            for item in raw_scores
            if isinstance(item, dict)
        )
        if isinstance(raw_scores, list)
        else ()
    )

    err_val = rec.get("error_code")
    text_val = rec.get("text_analysis")
    stack_val = rec.get("stack_excerpt")
    return EventRecord(
        run_id=str(rec.get("run_id") or default_run_id),
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        probe_id=probe_id,
        site=site,
        probe_type=probe_type,
        file_line=file_line,
        checks=checks,
        payload_json=payload_json,
        check_scores=scores,
        verdict=verdict,
        error_code=str(err_val) if err_val is not None else None,
        text_analysis=str(text_val) if text_val is not None else None,
        stack_excerpt=str(stack_val) if stack_val is not None else None,
        judged_at=judged_at,
        latency_ms=latency_ms,
    )


def _read_run_file(path: Path) -> tuple[RunRecord | None, list[EventRecord]]:
    if not path.is_file():
        return None, []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None, []

    run_id = path.stem
    started_at = ""
    finished_at: str | None = None
    cmd = ""
    pid = 0
    status = "running"
    halt_call_seq: int | None = None
    decision_model = ""
    text_model = ""
    has_start = False
    events_by_seq: dict[int, EventRecord] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue

        rec_type = rec.get("type")
        if rec_type == "run_start":
            has_start = True
            run_id = str(rec.get("id") or run_id)
            started_at = str(rec.get("started_at") or "")
            cmd = str(rec.get("cmd") or "")
            try:
                pid = int(rec.get("pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            status = str(rec.get("status") or "running")
            decision_model = str(rec.get("decision_model") or "")
            text_model = str(rec.get("text_model") or "")
        elif rec_type == "run_finish":
            if rec.get("status"):
                status = str(rec["status"])
            if rec.get("finished_at"):
                finished_at = str(rec["finished_at"])
            if halt_call_seq is None and rec.get("halt_call_seq") is not None:
                try:
                    halt_call_seq = int(rec["halt_call_seq"])
                except (TypeError, ValueError):
                    pass
        elif rec_type == "event":
            ev = _parse_event_record(rec, run_id)
            if ev is not None:
                events_by_seq[ev.call_seq] = ev

    if not has_start:
        return None, []

    events = sorted(events_by_seq.values(), key=lambda e: e.call_seq)
    pass_count = sum(1 for e in events if e.verdict == "pass")
    fail_count = sum(1 for e in events if e.verdict == "fail")
    error_count = sum(1 for e in events if e.verdict == "error")
    active_probes = len({e.probe_id for e in events})

    run = RunRecord(
        id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        cmd=cmd,
        pid=pid,
        status=status,
        halt_call_seq=halt_call_seq,
        decision_model=decision_model,
        text_model=text_model,
        pass_count=pass_count,
        fail_count=fail_count,
        error_count=error_count,
        active_probes=active_probes,
    )
    return run, events


class ProbeStore:
    """Thread-safe per-run JSONL store backing .probes/runs/<run_id>.jsonl."""

    def __init__(self, runs_dir: Path | str, *, create: bool = True) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()
        if create:
            _ensure_probes_gitignore(self.runs_dir)

    def close(self) -> None:
        return None

    def run_path(self, run_id: str) -> Path:
        if not run_id or not run_id.strip() or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return self.runs_dir / f"{run_id}.jsonl"

    def _append_record(self, run_id: str, record: dict[str, Any]) -> None:
        path = self.run_path(run_id)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    def create_run(
        self,
        run_id: str,
        *,
        started_at: str,
        cmd: str,
        pid: int,
        decision_model: str,
        text_model: str,
    ) -> None:
        self._append_record(
            run_id,
            {
                "type": "run_start",
                "id": run_id,
                "started_at": started_at,
                "cmd": cmd,
                "pid": pid,
                "status": "running",
                "decision_model": decision_model,
                "text_model": text_model,
            },
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: str | None = None,
        halt_call_seq: int | None = None,
    ) -> None:
        self._append_record(
            run_id,
            {
                "type": "run_finish",
                "id": run_id,
                "status": status,
                "finished_at": finished_at or utc_now_iso(),
                "halt_call_seq": halt_call_seq,
            },
        )

    def insert_event(
        self,
        *,
        run_id: str,
        call_seq: int,
        elapsed_ms: float,
        started_at: str,
        probe_id: str,
        site: str,
        probe_type: str,
        file_line: str,
        checks: Sequence[str],
        payload_json: str,
        check_scores: Sequence[CheckScore],
        verdict: str,
        error_code: str | None = None,
        text_analysis: str | None = None,
        stack_excerpt: str | None = None,
        judged_at: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        has_parsed = True
        try:
            parsed_payload: Any = json.loads(payload_json)
        except ValueError:
            has_parsed = False
            parsed_payload = payload_json

        rec: dict[str, Any] = {
            "type": "event",
            "run_id": run_id,
            "call_seq": call_seq,
            "elapsed_ms": elapsed_ms,
            "started_at": started_at,
            "probe_id": probe_id,
            "site": site,
            "probe_type": probe_type,
            "file_line": file_line,
            "checks": list(checks),
            "payload": parsed_payload,
            "check_scores": [
                {
                    "check": s.check,
                    "probability": s.probability,
                    "passed": s.passed,
                }
                for s in check_scores
            ],
            "verdict": verdict,
            "error_code": error_code,
            "text_analysis": text_analysis,
            "stack_excerpt": stack_excerpt,
            "judged_at": judged_at or utc_now_iso(),
            "latency_ms": latency_ms,
        }
        if not has_parsed:
            rec["payload_is_raw"] = True
        self._append_record(run_id, rec)

    def _iter_runs_sorted(self) -> list[tuple[RunRecord, Path]]:
        if not self.runs_dir.is_dir():
            return []
        found: list[tuple[RunRecord, Path]] = []
        for fp in sorted(self.runs_dir.glob("*.jsonl")):
            run, _ = _read_run_file(fp)
            if run is not None:
                found.append((run, fp))
        found.sort(key=lambda item: (item[0].started_at, item[0].id), reverse=True)
        return found

    def get_run(self, run_id: str | None = None) -> RunRecord | None:
        if run_id:
            try:
                path = self.run_path(run_id)
            except ValueError:
                return None
            run, _ = _read_run_file(path)
            return run
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        return [run for run, _ in self._iter_runs_sorted()[: max(1, limit)]]

    def list_events(
        self,
        run_id: str,
        *,
        probe_patterns: Sequence[str] | None = None,
    ) -> list[EventRecord]:
        """Return events for a run sorted strictly by call_seq ASC."""
        try:
            path = self.run_path(run_id)
        except ValueError:
            return []
        _, events = _read_run_file(path)
        patterns = [p for p in (probe_patterns or ()) if p]
        if not patterns:
            return events
        return [e for e in events if any(fnmatchcase(e.probe_id, pat) for pat in patterns)]

    def clean_runs(self, keep: int = 20) -> int:
        """Delete run logs older than the most recent `keep` runs."""
        keep_n = max(0, keep)
        ordered = self._iter_runs_sorted()
        deleted = 0
        for _, path in ordered[keep_n:]:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass
        return deleted
