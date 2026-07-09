import pytest
import xarray as xr
import trajan as ta


@pytest.fixture
def plot(pytestconfig):
    return pytestconfig.getoption('plot')

@pytest.fixture
def openoil():
    with xr.open_dataset(ta.DATA_DIR + 'openoil.nc') as ds:
        ds = ds.where(ds.status >= 0)  # to be removed
        yield ds

@pytest.fixture
def barents():
    with xr.open_dataset(ta.DATA_DIR + 'barents.nc') as ds:
        yield ds

@pytest.fixture
def parcels():
    with xr.open_dataset(ta.DATA_DIR + 'parcels.zarr', engine='zarr') as ds:
        yield ds

@pytest.fixture
def drifter_csv():
    fn = ta.DATA_DIR + 'omb/bug05_pos.csv'
    return fn
