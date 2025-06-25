from numpy.testing import assert_almost_equal
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd

def test_no_time_dim(barents):
    b = barents.drop_vars('drifter_names')
    b = b.isel(trajectory=0).mean(dim='obs')
    assert 'time' not in b.variables

    # can we use trajan without

    print(b.traj)

def test_custom_dims(barents: xr.DataArray):
    # b = barents.expand_dims('campaign')
    b = barents.rename(trajectory='campaign')
    print(b)

    g = b.traj(trajectory_dim='campaign').gridtime('1H')
    print(g)
