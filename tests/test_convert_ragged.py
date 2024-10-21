import xarray as xr
import trajan as _


def test_convert_ragged(test_data):
    ds = xr.open_dataset(test_data / 'xr_spotter_bulk_test_data.nc')
    trajified = ds.traj
