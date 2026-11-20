"""Pytest plugin for `manners`: --manners/--no-manners, @pytest.mark.{manners,no_manners}, and thread halt draining."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import manners
from manners import MannersHaltError, handle_thread_excepthook
from manners.config import has_legacy_probes_without_manners


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("manners")
    group.addoption(
        "--manners",
        action="store_const",
        const=True,
        dest="manners_cli_mode",
        default=None,
        help="Enable @manners and manners.here checks during pytest run (overrides --no-manners).",
    )
    group.addoption(
        "--no-manners",
        action="store_const",
        const=False,
        dest="manners_cli_mode",
        default=None,
        help=(
            "Disable @manners and manners.here checks by default during pytest run "
            "(override with @pytest.mark.manners or --manners)."
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "manners: allow @manners and manners.here checks on this test when --no-manners is set "
        "(pass force=True or --manners to enable even when MANNERS is unset)",
    )
    config.addinivalue_line(
        "markers",
        "no_manners: suppress @manners and manners.here checks for this test even when MANNERS=1 or --manners is set",
    )
    prev_hook = threading.excepthook
    config.add_cleanup(lambda: setattr(threading, "excepthook", prev_hook))
    threading.excepthook = lambda args: handle_thread_excepthook(args, prev_hook, suppress_forward=True)
    baseline = manners.get_active_runtime_stats()
    config._manners_plugin_state = {  # type: ignore[attr-defined]
        "marked_tests": 0,
        "baseline_run_id": baseline["run_id"] if baseline else None,
        "baseline_calls": baseline["calls"] if baseline else 0,
        "baseline_passed": baseline["passed"] if baseline else 0,
        "baseline_failed": baseline["failed"] if baseline else 0,
        "baseline_errors": baseline["errors"] if baseline else 0,
    }


def _resolve_item_manners_ctx(item: pytest.Item) -> contextlib.AbstractContextManager[None]:
    cli_mode = item.config.getoption("manners_cli_mode", default=None)
    if item.get_closest_marker("no_manners") is not None:
        return manners.disabled()
    manners_mark = item.get_closest_marker("manners")
    if manners_mark is not None:
        force = bool(manners_mark.kwargs.get("force", False))
        if force or cli_mode is True:
            return manners.enabled()
        return contextlib.nullcontext()
    if cli_mode is True:
        return manners.enabled()
    if cli_mode is False:
        return manners.disabled()
    return contextlib.nullcontext()


def _run_phase_with_manners(item: pytest.Item) -> Iterator[object]:
    with _resolve_item_manners_ctx(item):
        try:
            res = yield
        except BaseException:
            manners.flush(check=False)
            raise
        manners.flush(check=True)
        return res


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[object]:
    return (yield from _run_phase_with_manners(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[object]:
    if item.get_closest_marker("manners") is not None and item.get_closest_marker("no_manners") is None:
        state = getattr(item.config, "_manners_plugin_state", None)
        if isinstance(state, dict):
            state["marked_tests"] = int(state.get("marked_tests", 0)) + 1
    return (yield from _run_phase_with_manners(item))


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    del exitstatus
    state = getattr(config, "_manners_plugin_state", None) or {}
    marked = int(state.get("marked_tests", 0))
    stats = manners.get_active_runtime_stats()

    calls = passed = failed = errors = 0
    if stats is not None:
        same_run = stats["run_id"] == state.get("baseline_run_id")
        calls = stats["calls"] - (int(state.get("baseline_calls", 0)) if same_run else 0)
        passed = stats["passed"] - (int(state.get("baseline_passed", 0)) if same_run else 0)
        failed = stats["failed"] - (int(state.get("baseline_failed", 0)) if same_run else 0)
        errors = stats["errors"] - (int(state.get("baseline_errors", 0)) if same_run else 0)

    if calls > 0 and stats is not None:
        terminalreporter.write_line(
            f"manners: {calls} call(s) ({passed} pass, {failed} fail, {errors} error) — run {stats['run_id']}"
        )
    elif marked > 0:
        terminalreporter.write_line(
            f"manners: OFF — {marked} @pytest.mark.manners test(s) ran with 0 behavioral checks "
            "(set MANNERS=1 or pass --manners)",
            yellow=True,
        )

    if has_legacy_probes_without_manners(config.rootpath):
        terminalreporter.write_line(
            "[manners] WARNING: legacy PROBES=1 is set while MANNERS is unset; "
            "checks are OFF. Rename PROBES=1 to MANNERS=1 (v0.7.0 rebrand).",
            yellow=True,
        )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Iterator[object]:
    return (yield from _run_phase_with_manners(item))


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[object],
    request: pytest.SubRequest,
) -> Iterator[object]:
    try:
        return (yield)
    except MannersHaltError as exc:
        if fixturedef.cached_result is None:
            fixturedef.cached_result = (None, fixturedef.cache_key(request), (exc, exc.__traceback__))
        raise
