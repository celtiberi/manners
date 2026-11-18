"""Pytest plugin for `probe`: --probes/--no-probes, @pytest.mark.{probes,no_probes}, and thread halt draining."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import pytest

import probe
from probe import handle_thread_excepthook


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
        "probes: enable @probe and probe.here checks for this test even when --no-probes is set",
    )
    config.addinivalue_line(
        "markers",
        "no_probes: suppress @probe and probe.here checks for this test even when PROBES=1 or --probes is set",
    )
    prev_hook = threading.excepthook
    config.add_cleanup(lambda: setattr(threading, "excepthook", prev_hook))
    threading.excepthook = lambda args: handle_thread_excepthook(args, prev_hook, suppress_forward=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    cli_mode = item.config.getoption("probe_cli_mode", default=None)
    if item.get_closest_marker("no_probes") is not None:
        ctx: contextlib.AbstractContextManager[None] = probe.disabled()
    elif item.get_closest_marker("probes") is not None:
        ctx = probe.enabled()
    elif cli_mode is True:
        ctx = probe.enabled()
    elif cli_mode is False:
        ctx = probe.disabled()
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        outcome = yield
        has_exc = outcome.excinfo is not None
        probe.flush(check=not has_exc)
