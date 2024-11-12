from numpy.testing import assert_almost_equal
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd

def test_no_time_dim(barents):
    print(barents)
    b = barents.drop_vars('drifter_names')
    b = b.isel(trajectory=0).mean(dim='obs')
    assert 'time' not in b.variables

    # can we use trajan without

    print(b.traj)
