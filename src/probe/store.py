"""Thread-safe SQLite WAL storage for probe runs and call-ordered events."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from probe.backends import CheckScore
from probe.config import utc_now_iso

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    cmd TEXT NOT NULL,
    pid INTEGER NOT NULL,
    status TEXT NOT NULL,
    halt_call_seq INTEGER,
    decision_model TEXT NOT NULL,
    text_model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    call_seq INTEGER NOT NULL,
    elapsed_ms REAL NOT NULL,
    started_at TEXT NOT NULL,
    probe_id TEXT NOT NULL,
    site TEXT NOT NULL,
    probe_type TEXT NOT NULL,
    file_line TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    check_scores_json TEXT NOT NULL,
    verdict TEXT NOT NULL,
    error_code TEXT,
    text_analysis TEXT,
    stack_excerpt TEXT,
    judged_at TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    PRIMARY KEY (run_id, call_seq),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, call_seq);
CREATE INDEX IF NOT EXISTS idx_events_run_probe_seq ON events(run_id, probe_id, call_seq);
"""


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


def _ensure_probes_gitignore(db_path: Path) -> None:
    parent = db_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.name == ".probes":
        gi = parent / ".gitignore"
        if not gi.exists():
            gi.write_text("probes.db*\n", encoding="utf-8")


class ProbeStore:
    """Thread-safe SQLite WAL store backing .probes/probes.db."""

    def __init__(self, db_path: Path | str, *, create: bool = True) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        if create:
            _ensure_probes_gitignore(self.db_path)
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )
        else:
            ro_uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(
                ro_uri,
                uri=True,
                timeout=30.0,
                check_same_thread=False,
            )
            try:
                self._conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                self._conn.close()
                imm_uri = f"{self.db_path.resolve().as_uri()}?mode=ro&immutable=1"
                self._conn = sqlite3.connect(
                    imm_uri,
                    uri=True,
                    timeout=30.0,
                    check_same_thread=False,
                )
        self._conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                if create:
                    self._conn.execute("PRAGMA busy_timeout=30000")
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                    self._conn.execute("PRAGMA foreign_keys=ON")
                    cols = {
                        str(r["name"])
                        for r in self._conn.execute("PRAGMA table_info(events)").fetchall()
                    }
                    if cols and "when_cond" in cols:
                        self._conn.execute("DROP TABLE events")
                        try:
                            self._conn.execute("DELETE FROM runs")
                        except sqlite3.Error:
                            pass
                    self._conn.executescript(_SCHEMA_SQL)
                    self._conn.commit()
        except Exception:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

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
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO runs (
                        id, started_at, finished_at, cmd, pid, status,
                        halt_call_seq, decision_model, text_model
                    ) VALUES (?, ?, NULL, ?, ?, 'running', NULL, ?, ?)
                    """,
                    (run_id, started_at, cmd, pid, decision_model, text_model),
                )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: str | None = None,
        halt_call_seq: int | None = None,
    ) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE runs
                    SET status = ?,
                        finished_at = COALESCE(?, finished_at),
                        halt_call_seq = COALESCE(halt_call_seq, ?)
                    WHERE id = ?
                    """,
                    (status, finished_at or utc_now_iso(), halt_call_seq, run_id),
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
        checks_json = json.dumps(list(checks), ensure_ascii=False)
        scores_json = json.dumps(
            [
                {
                    "check": s.check,
                    "probability": s.probability,
                    "passed": s.passed,
                }
                for s in check_scores
            ],
            ensure_ascii=False,
        )
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO events (
                        run_id, call_seq, elapsed_ms, started_at, probe_id,
                        site, probe_type, file_line, checks_json,
                        payload_json, check_scores_json, verdict, error_code,
                        text_analysis, stack_excerpt, judged_at, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        call_seq,
                        elapsed_ms,
                        started_at,
                        probe_id,
                        site,
                        probe_type,
                        file_line,
                        checks_json,
                        payload_json,
                        scores_json,
                        verdict,
                        error_code,
                        text_analysis,
                        stack_excerpt,
                        judged_at or utc_now_iso(),
                        latency_ms,
                    ),
                )

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=str(row["id"]),
            started_at=str(row["started_at"]),
            finished_at=row["finished_at"],
            cmd=str(row["cmd"]),
            pid=int(row["pid"]),
            status=str(row["status"]),
            halt_call_seq=int(row["halt_call_seq"]) if row["halt_call_seq"] is not None else None,
            decision_model=str(row["decision_model"]),
            text_model=str(row["text_model"]),
            pass_count=int(row["pass_count"] or 0),
            fail_count=int(row["fail_count"] or 0),
            error_count=int(row["error_count"] or 0),
            active_probes=int(row["active_probes"] or 0),
        )

    def get_run(self, run_id: str | None = None) -> RunRecord | None:
        where = "WHERE r.id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        sql = f"""
            SELECT
                r.*,
                SUM(CASE WHEN e.verdict = 'pass' THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN e.verdict = 'fail' THEN 1 ELSE 0 END) AS fail_count,
                SUM(CASE WHEN e.verdict = 'error' THEN 1 ELSE 0 END) AS error_count,
                COUNT(DISTINCT e.probe_id) AS active_probes
            FROM runs r
            LEFT JOIN events e ON e.run_id = r.id
            {where}
            GROUP BY r.id
            ORDER BY r.started_at DESC, r.id DESC
            LIMIT 1
        """
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return self._row_to_run(row) if row is not None else None

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        sql = """
            SELECT
                r.*,
                SUM(CASE WHEN e.verdict = 'pass' THEN 1 ELSE 0 END) AS pass_count,
                SUM(CASE WHEN e.verdict = 'fail' THEN 1 ELSE 0 END) AS fail_count,
                SUM(CASE WHEN e.verdict = 'error' THEN 1 ELSE 0 END) AS error_count,
                COUNT(DISTINCT e.probe_id) AS active_probes
            FROM runs r
            LEFT JOIN events e ON e.run_id = r.id
            GROUP BY r.id
            ORDER BY r.started_at DESC, r.id DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, (max(1, limit),)).fetchall()
        return [self._row_to_run(r) for r in rows]

    def list_events(
        self,
        run_id: str,
        *,
        probe_patterns: Sequence[str] | None = None,
    ) -> list[EventRecord]:
        """Return events for a run sorted strictly by call_seq ASC."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM events
                WHERE run_id = ?
                ORDER BY call_seq ASC
                """,
                (run_id,),
            ).fetchall()

        patterns = [p for p in (probe_patterns or ()) if p]
        events: list[EventRecord] = []
        for row in rows:
            pid = str(row["probe_id"])
            if patterns and not any(fnmatchcase(pid, pat) for pat in patterns):
                continue
            try:
                raw_checks = json.loads(row["checks_json"])
                checks = tuple(str(c) for c in raw_checks) if isinstance(raw_checks, list) else ()
            except Exception:
                checks = ()
            try:
                scores_raw = json.loads(row["check_scores_json"])
                scores = (
                    tuple(
                        CheckScore(
                            check=str(item.get("check", "")),
                            probability=float(item.get("probability", 0.0)),
                            passed=bool(item.get("passed", False)),
                        )
                        for item in scores_raw
                        if isinstance(item, dict)
                    )
                    if isinstance(scores_raw, list)
                    else ()
                )
            except Exception:
                scores = ()
            events.append(
                EventRecord(
                    run_id=str(row["run_id"]),
                    call_seq=int(row["call_seq"]),
                    elapsed_ms=float(row["elapsed_ms"]),
                    started_at=str(row["started_at"]),
                    probe_id=pid,
                    site=str(row["site"]),
                    probe_type=str(row["probe_type"]),
                    file_line=str(row["file_line"]),
                    checks=checks,
                    payload_json=str(row["payload_json"]),
                    check_scores=scores,
                    verdict=str(row["verdict"]),
                    error_code=row["error_code"],
                    text_analysis=row["text_analysis"],
                    stack_excerpt=row["stack_excerpt"],
                    judged_at=str(row["judged_at"]),
                    latency_ms=float(row["latency_ms"]),
                )
            )
        return events

    def clean_runs(self, keep: int = 20) -> int:
        """Delete runs older than the most recent `keep` runs and truncate the WAL."""
        keep_n = max(0, keep)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    DELETE FROM events
                    WHERE run_id NOT IN (
                        SELECT id FROM runs ORDER BY started_at DESC, id DESC LIMIT ?
                    )
                    """,
                    (keep_n,),
                )
                cur = self._conn.execute(
                    """
                    DELETE FROM runs
                    WHERE id NOT IN (
                        SELECT id FROM runs ORDER BY started_at DESC, id DESC LIMIT ?
                    )
                    """,
                    (keep_n,),
                )
                deleted = int(cur.rowcount or 0)
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            return deleted
