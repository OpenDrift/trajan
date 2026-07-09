import trajan as ta
import xarray as xr

def test_no_time_dim(barents):
    b = barents.drop_vars('drifter_names')
    b = b.isel(trajectory=0).mean(dim='obs')
    assert 'time' not in b.variables

    # can we use trajan without

    print(b.traj)
