from numpy.testing import assert_almost_equal
import pytest
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd


def test_to2d(barents):
    # print(barents)
    gr = barents.traj.gridtime('1h')
    # print(gr)

    assert gr.traj.is_1d()

    b2d = gr.traj.to_2d()
    assert b2d.traj.is_2d()


def test_to1d(barents):
    # print(barents)
    gr = barents.traj.gridtime('1h')
    assert gr.traj.is_1d()

    gr = gr.traj.to_1d()
    assert gr.traj.is_1d()

    with pytest.raises(ValueError):
        barents.traj.to_1d()

    print('converting to 1d')
    gr = barents.isel(trajectory=0).traj.to_1d()
    assert gr.traj.is_1d()


def test_trajectories_group(barents):
    assert barents.traj.is_2d()
    # print(barents)

    def a(t):
        print(t)
        return t.mean('obs')

    # Calculates mean of each trajectory
    p = barents.traj.trajectories().map(a)
    print(p)
