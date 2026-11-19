"""Thread-safe per-run JSONL storage for manners runs and call-ordered events."""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from manners.backends import CheckScore
from manners.config import utc_now_iso

_RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(slots=True, frozen=True)
class RunRecord:
    """Summary metadata and verdict counts for a single checked process run."""

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
    active_checks: int = 0


@dataclass(slots=True, frozen=True)
class EventRecord:
    """Single evaluated manners call ordered by call_seq."""

    run_id: str
    call_seq: int
    elapsed_ms: float
    started_at: str
    check_id: str
    site: str
    check_type: str
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


def _ensure_manners_gitignore(runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(runs_dir, 0o700)
    except OSError:
        pass
    inner_gi = runs_dir / ".gitignore"
    if not inner_gi.exists():
        try:
            inner_gi.write_text("*\n!.gitignore\n", encoding="utf-8")
        except OSError:
            pass
    parent = runs_dir.parent
    if parent.name == ".manners":
        gi = parent / ".gitignore"
        if not gi.exists():
            try:
                gi.write_text("runs/\n", encoding="utf-8")
            except OSError:
                pass


def _finite_float(val: Any, default: float | None = None) -> float:
    if isinstance(val, bool):
        raise ValueError(f"Expected numeric float, got bool {val!r}")
    if val is None:
        if default is None:
            raise ValueError("Missing required numeric value")
        return default
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"Non-finite float value: {val!r}")
    return fval


def _parse_event_record(rec: dict[str, Any], default_run_id: str) -> EventRecord | None:
    try:
        call_seq = int(rec["call_seq"])
        elapsed_ms = _finite_float(rec["elapsed_ms"])
        started_at = str(rec["started_at"])
        check_id = str(rec["check_id"])
        site = str(rec["site"])
        check_type = str(rec["check_type"])
        file_line = str(rec["file_line"])
        verdict = str(rec["verdict"])
        judged_at = str(rec.get("judged_at") or started_at)
        latency_ms = _finite_float(rec.get("latency_ms"), default=0.0)

        raw_scores = rec.get("check_scores")
        scores_list: list[CheckScore] = []
        if isinstance(raw_scores, list):
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                scores_list.append(
                    CheckScore(
                        check=str(item.get("check", "")),
                        probability=_finite_float(item.get("probability"), default=0.0),
                        passed=bool(item.get("passed", False)),
                    )
                )
        scores = tuple(scores_list)
    except (KeyError, TypeError, ValueError):
        return None

    raw_checks = rec.get("checks")
    checks = tuple(str(c) for c in raw_checks) if isinstance(raw_checks, list) else ()

    if rec.get("payload_is_raw"):
        payload_json = str(rec.get("payload", ""))
    else:
        payload_json = json.dumps(rec.get("payload"), ensure_ascii=False)

    err_val = rec.get("error_code")
    text_val = rec.get("text_analysis")
    stack_val = rec.get("stack_excerpt")
    return EventRecord(
        run_id=default_run_id,
        call_seq=call_seq,
        elapsed_ms=elapsed_ms,
        started_at=started_at,
        check_id=check_id,
        site=site,
        check_type=check_type,
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


def _peek_run_start(path: Path) -> str | None:
    """Read just the run_start timestamp from a run file without parsing event payloads."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("type") == "run_start":
                    return str(rec.get("started_at") or "")
    except (OSError, UnicodeDecodeError):
        return None
    return None


_STATUS_PRIORITY: dict[str, int] = {
    "running": 0,
    "completed": 1,
    "failed": 2,
    "halted": 3,
}


def _read_run_file(path: Path) -> tuple[RunRecord | None, list[EventRecord]]:
    if not path.is_file():
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

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
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
                    new_status = rec.get("status")
                    if new_status:
                        new_status_str = str(new_status)
                        if _STATUS_PRIORITY.get(new_status_str, 1) >= _STATUS_PRIORITY.get(status, 0):
                            status = new_status_str
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
    except (OSError, UnicodeDecodeError):
        return None, []

    if not has_start:
        return None, []

    events = sorted(events_by_seq.values(), key=lambda e: e.call_seq)
    pass_count = sum(1 for e in events if e.verdict == "pass")
    fail_count = sum(1 for e in events if e.verdict == "fail")
    error_count = sum(1 for e in events if e.verdict == "error")
    active_checks = len({e.check_id for e in events})

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
        active_checks=active_checks,
    )
    return run, events


class MannersStore:
    """Thread-safe per-run JSONL store backing .manners/runs/<run_id>.jsonl."""

    def __init__(self, runs_dir: Path | str, *, create: bool = True) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()
        if create:
            _ensure_manners_gitignore(self.runs_dir)

    def close(self) -> None:
        return None

    def run_path(self, run_id: str) -> Path:
        if (
            not run_id
            or not _RUN_ID_RE.fullmatch(run_id)
            or ".." in run_id
        ):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        candidate = self.runs_dir / f"{run_id}.jsonl"
        if candidate.resolve().parent != self.runs_dir.resolve():
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return candidate

    def _append_record(self, run_id: str, record: dict[str, Any]) -> None:
        path = self.run_path(run_id)
        data = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            needs_leading_newline = False
            try:
                if path.exists() and path.stat().st_size > 0:
                    with path.open("rb") as rfh:
                        rfh.seek(-1, os.SEEK_END)
                        needs_leading_newline = rfh.read(1) != b"\n"
            except OSError:
                pass
            if needs_leading_newline:
                data = b"\n" + data

            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                try:
                    os.fchmod(fd, 0o600)
                except OSError:
                    pass
                view = memoryview(data)
                written = 0
                while written < len(data):
                    written += os.write(fd, view[written:])
            finally:
                os.close(fd)

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
        check_id: str,
        site: str,
        check_type: str,
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
            "check_id": check_id,
            "site": site,
            "check_type": check_type,
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

    def _ordered_run_files(self) -> tuple[list[tuple[str, str, Path]], list[Path]]:
        if not self.runs_dir.is_dir():
            return [], []
        valid: list[tuple[str, str, Path]] = []
        orphans: list[Path] = []
        for fp in sorted(self.runs_dir.glob("*.jsonl")):
            started_at = _peek_run_start(fp)
            if started_at is None:
                orphans.append(fp)
            else:
                valid.append((started_at, fp.stem, fp))
        valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return valid, orphans

    def get_run_with_events(self, run_id: str | None = None) -> tuple[RunRecord | None, list[EventRecord]]:
        if run_id:
            try:
                path = self.run_path(run_id)
            except ValueError:
                return None, []
            return _read_run_file(path)
        valid, _ = self._ordered_run_files()
        for _, _, fp in valid:
            run, events = _read_run_file(fp)
            if run is not None:
                return run, events
        return None, []

    def get_run(self, run_id: str | None = None) -> RunRecord | None:
        run, _ = self.get_run_with_events(run_id)
        return run

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        valid, _ = self._ordered_run_files()
        out: list[RunRecord] = []
        cap = max(1, limit)
        for _, _, fp in valid:
            if len(out) >= cap:
                break
            run, _ = _read_run_file(fp)
            if run is not None:
                out.append(run)
        return out

    def list_events(
        self,
        run_id: str,
        *,
        check_patterns: Sequence[str] | None = None,
    ) -> list[EventRecord]:
        """Return events for a run sorted strictly by call_seq ASC."""
        try:
            path = self.run_path(run_id)
        except ValueError:
            return []
        _, events = _read_run_file(path)
        patterns = [p for p in (check_patterns or ()) if p]
        if not patterns:
            return events
        return [e for e in events if any(fnmatchcase(e.check_id, pat) for pat in patterns)]

    def clean_runs(self, keep: int = 20) -> int:
        """Delete run logs older than the most recent `keep` runs."""
        keep_n = max(0, keep)
        valid, orphans = self._ordered_run_files()
        to_delete = [fp for _, _, fp in valid[keep_n:]] + orphans
        deleted = 0
        for path in to_delete:
            if path.exists():
                path.unlink()
                deleted += 1
        return deleted
