"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

CYCLING_INDOOR    = FIXTURES / "cycling_indoor.fit"
RUNNING_TREADMILL = FIXTURES / "running_treadmill.fit"
RUNNING_OUTDOOR   = FIXTURES / "running_outdoor_marathon.fit"


@pytest.fixture(scope="session")
def cycling_fit():
    return str(CYCLING_INDOOR)


@pytest.fixture(scope="session")
def treadmill_fit():
    return str(RUNNING_TREADMILL)


@pytest.fixture(scope="session")
def marathon_fit():
    return str(RUNNING_OUTDOOR)


@pytest.fixture(scope="session")
def cycling_df(cycling_fit):
    from fit_analyser.parser import parse_fit_to_dataframe
    return parse_fit_to_dataframe(cycling_fit)


@pytest.fixture(scope="session")
def treadmill_df(treadmill_fit):
    from fit_analyser.parser import parse_fit_to_dataframe
    return parse_fit_to_dataframe(treadmill_fit)


@pytest.fixture(scope="session")
def marathon_df(marathon_fit):
    from fit_analyser.parser import parse_fit_to_dataframe
    return parse_fit_to_dataframe(marathon_fit)


@pytest.fixture(scope="session")
def cycling_meta(cycling_fit):
    from fit_analyser.parser import get_session_meta
    return get_session_meta(cycling_fit)


@pytest.fixture(scope="session")
def treadmill_meta(treadmill_fit):
    from fit_analyser.parser import get_session_meta
    return get_session_meta(treadmill_fit)


@pytest.fixture(scope="session")
def marathon_meta(marathon_fit):
    from fit_analyser.parser import get_session_meta
    return get_session_meta(marathon_fit)
