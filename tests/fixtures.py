import pytest
import lzma
import xarray as xr
from pathlib import Path

@pytest.fixture
def plot(pytestconfig):
    return pytestconfig.getoption('plot')

@pytest.fixture(scope='session')
def opendrift_sim():
    oil = Path(__file__).parent.parent / 'examples' / 'openoil.nc.xz'
    with lzma.open(oil) as oil:
        ds = xr.open_dataset(oil)
        ds.load()
        return ds
