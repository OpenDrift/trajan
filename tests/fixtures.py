import pytest
import lzma
import xarray as xr
from pathlib import Path

@pytest.fixture
def plot(pytestconfig):
    return pytestconfig.getoption('plot')

@pytest.fixture
def opendrift_sim():
    oil = Path(__file__).parent.parent / 'examples' / 'openoil.nc.xz'
    with lzma.open(oil) as oil:
        ds = xr.open_dataset(oil)
        ds.load()
        return ds

@pytest.fixture
def barents():
    fn = Path(__file__).parent.parent / 'examples' / 'barents.nc.xz'
    with lzma.open(fn) as b:
        ds = xr.open_dataset(b)
        ds.load()
        return ds

@pytest.fixture
def drifter_csv():
    fn = Path(__file__).parent.parent / 'examples' / 'bug05_pos.csv.xz'
    return fn
