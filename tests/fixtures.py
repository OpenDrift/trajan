import pytest
import os

@pytest.fixture
def show_plot(pytestconfig):
    return pytestconfig.getoption('plot')

