"""Pytest plugin for `probe`: --probes/--no-probes, @pytest.mark.{probes,no_probes}, and thread halt draining."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import pytest

import probe
from probe import ProbeHaltError, handle_thread_excepthook


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("probe")
    group.addoption(
        "--probes",
        action="store_const",
        const=True,
        dest="probe_cli_mode",
        default=None,
        help="Enable @probe and probe.here checks during pytest run (overrides --no-probes).",
    )
    group.addoption(
        "--no-probes",
        action="store_const",
        const=False,
        dest="probe_cli_mode",
        default=None,
        help=(
            "Disable @probe and probe.here checks by default during pytest run "
            "(override with @pytest.mark.probes or --probes)."
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "probes: allow @probe and probe.here checks on this test when --no-probes is set "
        "(pass force=True or --probes to enable even when PROBES is unset)",
    )
    config.addinivalue_line(
        "markers",
        "no_probes: suppress @probe and probe.here checks for this test even when PROBES=1 or --probes is set",
    )
    prev_hook = threading.excepthook
    config.add_cleanup(lambda: setattr(threading, "excepthook", prev_hook))
    threading.excepthook = lambda args: handle_thread_excepthook(args, prev_hook, suppress_forward=True)


def _resolve_item_probe_ctx(item: pytest.Item) -> contextlib.AbstractContextManager[None]:
    cli_mode = item.config.getoption("probe_cli_mode", default=None)
    if item.get_closest_marker("no_probes") is not None:
        return probe.disabled()
    probes_mark = item.get_closest_marker("probes")
    if probes_mark is not None:
        force = bool(probes_mark.kwargs.get("force", False))
        if force or cli_mode is True:
            return probe.enabled()
        return contextlib.nullcontext()
    if cli_mode is True:
        return probe.enabled()
    if cli_mode is False:
        return probe.disabled()
    return contextlib.nullcontext()


def _run_phase_with_probes(item: pytest.Item) -> Iterator[object]:
    with _resolve_item_probe_ctx(item):
        try:
            res = yield
        except BaseException:
            probe.flush(check=False)
            raise
        probe.flush(check=True)
        return res


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Iterator[object]:
    return (yield from _run_phase_with_probes(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[object]:
    return (yield from _run_phase_with_probes(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Iterator[object]:
    return (yield from _run_phase_with_probes(item))


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[object],
    request: pytest.SubRequest,
) -> Iterator[object]:
    try:
        return (yield)
    except ProbeHaltError as exc:
        if fixturedef.cached_result is None:
            fixturedef.cached_result = (None, fixturedef.cache_key(request), (exc, exc.__traceback__))
        raise
