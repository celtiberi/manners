"""Tests for stackable manners.enabled()/disabled() and the built-in pytest11 plugin."""

from __future__ import annotations

import concurrent.futures
import http.server
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import manners
from manners.config import clear_env_cache, is_manners_enabled


def test_stackable_enabled_and_disabled_scopes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANNERS", raising=False)
    clear_env_cache()

    assert is_manners_enabled(tmp_path) is False

    # Process-wide stack: enabled() -> disabled() -> enabled() works across ThreadPoolExecutor workers.
    def worker_check(_idx: int) -> bool:
        return is_manners_enabled(tmp_path)

    with manners.enabled():
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(worker_check, [1, 2])) == [True, True]
        with manners.disabled():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                assert list(pool.map(worker_check, [1, 2])) == [False, False]
            with manners.enabled():
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    assert list(pool.map(worker_check, [1, 2])) == [True, True]
            assert is_manners_enabled(tmp_path) is False
        assert is_manners_enabled(tmp_path) is True
    assert is_manners_enabled(tmp_path) is False

    # Context scope stays local to current context without changing worker threads.
    with manners.enabled(scope="context"):
        assert is_manners_enabled(tmp_path) is True
        with manners.disabled(scope="context"):
            assert is_manners_enabled(tmp_path) is False
        assert is_manners_enabled(tmp_path) is True
    assert is_manners_enabled(tmp_path) is False

    with pytest.raises(ValueError, match="Invalid enabled scope"):
        with manners.enabled(scope="bogus"):
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
        (proj / ".env").write_text("MANNERS=1\nMANNERS_NO_PDB=1\n", encoding="utf-8")
        (proj / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [project]
                name = "plugin-subproc"
                version = "0.1.0"

                [tool.pytest.ini_options]
                addopts = "--no-manners"

                [tool.manners.models.decision]
                provider = "http_decision"
                url = "http://127.0.0.1:{port}/decision"
                model = "jev"

                [tool.manners.text]
                explain_decision_failures = false

                [tool.manners.check]
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
                from manners import manners


                def test_mocked_unit_skips_under_no_manners():
                    manners.here("Output is GOOD", id="sub.unit", reply="BAD canned stub")


                @pytest.mark.manners
                def test_manners_marker_overrides_no_manners():
                    manners.here("Output is GOOD", id="sub.live_ok", reply="GOOD verified reply")


                @pytest.mark.no_manners
                def test_no_manners_marker_wins_over_cli_flag():
                    manners.here("Output is GOOD", id="sub.no_manners", reply="BAD canned stub")


                @pytest.mark.manners
                def test_daemon_thread_halt_fails_test():
                    in_call = threading.Event()

                    def bg() -> None:
                        in_call.set()
                        time.sleep(0.02)
                        manners.here("Output is GOOD", id="sub.daemon_bad", reply="BAD async failure")

                    t = threading.Thread(target=bg, daemon=True)
                    t.start()
                    assert in_call.wait(timeout=2.0)
                    time.sleep(0.15)


                @pytest.fixture
                def mocked_setup_teardown_fixture():
                    manners.here("Output is GOOD", id="sub.fix_setup", reply="BAD setup stub")
                    yield "ok"
                    manners.here("Output is GOOD", id="sub.fix_teardown", reply="BAD teardown stub")


                def test_fixture_setup_and_teardown_skip_under_no_manners(mocked_setup_teardown_fixture):
                    assert mocked_setup_teardown_fixture == "ok"


                @pytest.fixture
                def failing_setup_fixture():
                    manners.here("Output is GOOD", id="sub.shared_fix", reply="BAD shared setup")
                    return "never"


                def test_shared_failing_fixture_first(failing_setup_fixture):
                    assert failing_setup_fixture == "never"


                def test_shared_failing_fixture_second(failing_setup_fixture):
                    assert failing_setup_fixture == "never"


                @pytest.mark.manners(force=True)
                def test_forced_manners_marker_runs_even_without_env():
                    manners.here("Output is GOOD", id="sub.forced_mark", reply="GOOD forced")
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        # 1. Under addopts="--no-manners": unmarked + fixture setup/teardown skip HTTP; @pytest.mark.manners runs.
        res_ok = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-k",
                "test_mocked_unit_skips_under_no_manners or test_manners_marker_overrides_no_manners "
                "or test_no_manners_marker_wins_over_cli_flag "
                "or test_fixture_setup_and_teardown_skip_under_no_manners",
                "-q",
            ],
            cwd=proj,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_ok.returncode == 0, res_ok.stdout + res_ok.stderr
        assert len(seen_snapshots) == 1
        assert "GOOD verified reply" in seen_snapshots[0]

        # 2. Passing --manners overrides addopts="--no-manners", while @pytest.mark.no_manners still wins!
        seen_snapshots.clear()
        res_cli = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--manners",
                "-k",
                "test_mocked_unit_skips_under_no_manners or test_no_manners_marker_wins_over_cli_flag",
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
        assert "sub.no_manners" not in res_cli.stdout + res_cli.stderr

        # 3. Unjoined daemon thread MannersHaltError fails with zero Pluggy/Thread warnings even when treated as errors.
        seen_snapshots.clear()
        bad_request_seen.clear()
        res_daemon = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-W",
                "error::pluggy.PluggyTeardownRaisedWarning",
                "-W",
                "error::pytest.PytestUnhandledThreadExceptionWarning",
                "-k",
                "test_daemon_thread_halt_fails_test",
                "-q",
            ],
            cwd=proj,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_daemon.returncode == 1
        combined = res_daemon.stdout + res_daemon.stderr
        assert "sub.daemon_bad" in combined
        assert "PluggyTeardownRaisedWarning" not in combined
        assert "PytestUnhandledThreadExceptionWarning" not in combined

        # 4. Shared fixture failing with MannersHaltError during setup cleanly fails both tests
        #    without _finalizers crash.
        seen_snapshots.clear()
        res_shared = subprocess.run(
            [sys.executable, "-m", "pytest", "--manners", "-k", "test_shared_failing_fixture_", "-q"],
            cwd=proj,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_shared.returncode == 1
        shared_out = res_shared.stdout + res_shared.stderr
        assert "2 errors" in shared_out
        assert "_finalizers" not in shared_out

        # 5. Explicit MANNERS=0 in env suppresses @pytest.mark.manners (honors env override cleanly).
        seen_snapshots.clear()
        env_off = {**os.environ, "MANNERS": "0"}
        res_env_off = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_manners_marker_overrides_no_manners", "-q"],
            cwd=proj,
            env=env_off,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_env_off.returncode == 0, res_env_off.stdout + res_env_off.stderr
        assert seen_snapshots == []

        # 6. Keyless/offline CI (no .env, MANNERS unset): @pytest.mark.manners skips;
        #    --manners and force=True still run.
        (proj / ".env").unlink()
        seen_snapshots.clear()
        env_ci = {k: v for k, v in os.environ.items() if k != "MANNERS"}
        res_ci = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_manners_marker_overrides_no_manners", "-q"],
            cwd=proj,
            env=env_ci,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_ci.returncode == 0, res_ci.stdout + res_ci.stderr
        assert seen_snapshots == []

        res_forced = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_forced_manners_marker_runs_even_without_env", "-q"],
            cwd=proj,
            env=env_ci,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_forced.returncode == 0, res_forced.stdout + res_forced.stderr
        assert len(seen_snapshots) == 1
        assert "GOOD forced" in seen_snapshots[0]
    finally:
        server.shutdown()
        server.server_close()


def test_v071_pytest_terminal_summary_active_vs_off(tmp_path: Path) -> None:
    class _DecisionHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            probs = [0.95 for _ in body.get("checks", [])]
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
        (proj / ".env").write_text("PROBES=1\nMANNERS_NO_PDB=1\n", encoding="utf-8")
        (proj / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [project]
                name = "summary-subproc"
                version = "0.1.0"

                [tool.pytest.ini_options]
                addopts = "--no-manners"

                [tool.manners.models.decision]
                provider = "http_decision"
                url = "http://127.0.0.1:{port}/decision"
                model = "jev"

                [tool.manners.text]
                explain_decision_failures = false

                [tool.manners.check]
                require_check = false
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (proj / "test_summary.py").write_text(
            textwrap.dedent(
                """
                import pytest
                from manners import manners

                def test_plain_unit():
                    manners.here("Output is GOOD", id="sum.unit", reply="GOOD")

                @pytest.mark.manners
                def test_marked_live():
                    manners.here("Output is GOOD", id="sum.live", reply="GOOD")
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        env_base = {k: v for k, v in os.environ.items() if k not in ("MANNERS", "PROBES")}

        # 1. Plain unmarked offline test under --no-manners prints no manners summary line.
        res_plain = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_plain_unit", "-q"],
            cwd=proj,
            env=env_base,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_plain.returncode == 0, res_plain.stdout + res_plain.stderr
        assert "manners:" not in res_plain.stdout

        # 2. @pytest.mark.manners with legacy PROBES=1 and MANNERS unset warns in pytest summary and prints OFF.
        res_off = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_marked_live", "-q"],
            cwd=proj,
            env=env_base,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_off.returncode == 0, res_off.stdout + res_off.stderr
        assert "manners: OFF — 1 @pytest.mark.manners test(s) ran with 0 behavioral checks" in res_off.stdout
        assert "legacy PROBES=1 is set while MANNERS is unset" in res_off.stdout

        # 3. With MANNERS=1 (-s uncaptured), active banner prints on stderr and call counts print in pytest summary.
        res_on = subprocess.run(
            [sys.executable, "-m", "pytest", "-k", "test_marked_live", "-q", "-s"],
            cwd=proj,
            env={**env_base, "MANNERS": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_on.returncode == 0, res_on.stdout + res_on.stderr
        assert "manners: 1 call(s) (1 pass, 0 fail, 0 error) — run " in res_on.stdout
        assert "[manners] active" in res_on.stderr
    finally:
        server.shutdown()
        server.server_close()
