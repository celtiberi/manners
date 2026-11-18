"""Tests for stackable probe.enabled()/disabled() and the built-in pytest11 plugin."""

from __future__ import annotations

import concurrent.futures
import http.server
import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import probe
from probe.config import clear_env_cache, is_probes_enabled


def test_stackable_enabled_and_disabled_scopes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROBES", raising=False)
    clear_env_cache()

    assert is_probes_enabled(tmp_path) is False

    # Process-wide stack: enabled() -> disabled() -> enabled() works across ThreadPoolExecutor workers.
    def worker_check(_idx: int) -> bool:
        return is_probes_enabled(tmp_path)

    with probe.enabled():
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(worker_check, [1, 2])) == [True, True]
        with probe.disabled():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                assert list(pool.map(worker_check, [1, 2])) == [False, False]
            with probe.enabled():
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    assert list(pool.map(worker_check, [1, 2])) == [True, True]
            assert is_probes_enabled(tmp_path) is False
        assert is_probes_enabled(tmp_path) is True
    assert is_probes_enabled(tmp_path) is False

    # Context scope stays local to current context without changing worker threads.
    with probe.enabled(scope="context"):
        assert is_probes_enabled(tmp_path) is True
        with probe.disabled(scope="context"):
            assert is_probes_enabled(tmp_path) is False
        assert is_probes_enabled(tmp_path) is True
    assert is_probes_enabled(tmp_path) is False

    with pytest.raises(ValueError, match="Invalid enabled scope"):
        with probe.enabled(scope="bogus"):
            pass


def test_pytest_plugin_markers_flags_and_daemon_thread_halt(tmp_path: Path) -> None:
    seen_snapshots: list[str] = []
    seen_lock = threading.Lock()
    bad_request_seen = threading.Event()

    class _DecisionHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw)
            snap = str(body.get("snapshot", ""))
            with seen_lock:
                seen_snapshots.append(snap)
            if "BAD" in snap:
                bad_request_seen.set()
            probs = [0.95 if "GOOD" in snap else 0.05 for _ in body.get("checks", [])]
            payload = json.dumps({"probabilities": probs}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _DecisionHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    try:
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / ".env").write_text("PROBES=1\nPROBE_NO_PDB=1\n", encoding="utf-8")
        (proj / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [project]
                name = "plugin-subproc"
                version = "0.1.0"

                [tool.pytest.ini_options]
                addopts = "--no-probes"

                [tool.probe.models.decision]
                provider = "http_decision"
                url = "http://127.0.0.1:{port}/decision"
                model = "jev"

                [tool.probe.text]
                explain_decision_failures = false

                [tool.probe.check]
                require_check = false
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (proj / "test_suite.py").write_text(
            textwrap.dedent(
                """
                import threading
                import time
                import pytest
                from probe import probe


                def test_mocked_unit_skips_under_no_probes():
                    probe.here("Output is GOOD", id="sub.unit", reply="BAD canned stub")


                @pytest.mark.probes
                def test_probed_marker_overrides_no_probes():
                    probe.here("Output is GOOD", id="sub.live_ok", reply="GOOD verified reply")


                @pytest.mark.no_probes
                def test_no_probes_marker_wins_over_cli_flag():
                    probe.here("Output is GOOD", id="sub.no_probes", reply="BAD canned stub")


                @pytest.mark.probes
                def test_daemon_thread_halt_fails_test():
                    in_call = threading.Event()

                    def bg() -> None:
                        in_call.set()
                        time.sleep(0.02)
                        probe.here("Output is GOOD", id="sub.daemon_bad", reply="BAD async failure")

                    t = threading.Thread(target=bg, daemon=True)
                    t.start()
                    assert in_call.wait(timeout=2.0)
                    time.sleep(0.15)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        # 1. Under addopts="--no-probes": unmarked + no_probes skip HTTP; @pytest.mark.probes runs HTTP and passes.
        res_ok = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "not test_daemon_thread_halt_fails_test", "-q"],
            cwd=proj,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_ok.returncode == 0, res_ok.stdout + res_ok.stderr
        assert len(seen_snapshots) == 1
        assert "GOOD verified reply" in seen_snapshots[0]

        # 2. Passing --probes overrides addopts="--no-probes", while @pytest.mark.no_probes still wins!
        seen_snapshots.clear()
        res_cli = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--probes",
                "-k",
                "test_mocked_unit_skips_under_no_probes or test_no_probes_marker_wins_over_cli_flag",
                "-q",
            ],
            cwd=proj,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_cli.returncode == 1
        assert len(seen_snapshots) == 1
        assert "sub.unit" in res_cli.stdout + res_cli.stderr
        assert "sub.no_probes" not in res_cli.stdout + res_cli.stderr

        # 3. Unjoined daemon thread ProbeHaltError is drained at test teardown without warning noise and fails test.
        seen_snapshots.clear()
        bad_request_seen.clear()
        res_daemon = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_daemon_thread_halt_fails_test", "-q"],
            cwd=proj,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_daemon.returncode == 1
        combined = res_daemon.stdout + res_daemon.stderr
        assert "sub.daemon_bad" in combined
        assert "PytestUnhandledThreadExceptionWarning" not in combined
    finally:
        server.shutdown()
        server.server_close()
