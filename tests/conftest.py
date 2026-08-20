from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--golden", action="store", default=None, help="external golden manifest")


@pytest.fixture
def golden_option(request: pytest.FixtureRequest):
    return request.config.getoption("--golden")

