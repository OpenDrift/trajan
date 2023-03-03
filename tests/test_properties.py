import pytest
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd

@pytest.mark.parametrize('ds', ['barents', 'opendrift_sim'])
def test_timestep(ds, request):
    ds = request.getfixturevalue(ds)
    print(ds.traj.timestep())

