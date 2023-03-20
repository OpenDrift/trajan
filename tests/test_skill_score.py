import xarray as xr
import trajan as _
import numpy as np


def test_barents_self(barents):
    # this should be full score..
    barents = barents.traj.gridtime('1H')
    print(barents)
    skill = barents.traj.skill(barents)
    print(skill)

    np.testing.assert_allclose(skill.values, 1.)

def test_barents_trajs(barents):
    barents = barents.traj.gridtime('1H')
    b0 = barents.isel(trajectory=0).dropna('time')
    b1 = barents.isel(trajectory=1).sel(time=slice(b0.time[0], b0.time[-1]))
    b1 = b1.traj.gridtime(b0.time)
    skill = b0.traj.skill(b1, tolerance_threshold=100)
    print(skill)
    np.testing.assert_allclose(skill.values, 0.)
