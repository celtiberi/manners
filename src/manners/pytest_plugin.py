"""Pytest plugin for `manners`: --manners/--no-manners, @pytest.mark.{manners,no_manners}, and thread halt draining."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import pytest

import manners
from manners import MannersHaltError, handle_thread_excepthook


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
    return (yield from _run_phase_with_manners(item))


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
