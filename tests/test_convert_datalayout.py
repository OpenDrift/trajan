from numpy.testing import assert_almost_equal
import pytest
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd
from trajan.readers.omb import read_omb_csv


def test_to_ragged(barents):
    # print(barents)
    gr = barents.traj.gridtime('1h')
    # print(gr)

    assert gr.traj.is_orthogonal()

    b_ragged = gr.traj.to_ragged()
    assert b_ragged.traj.is_ragged()


def test_to_orthogonal(barents):
    # print(barents)
    gr = barents.traj.gridtime('1h')
    assert gr.traj.is_orthogonal()

    gr = gr.traj.to_orthogonal()
    assert gr.traj.is_orthogonal()

    with pytest.raises(ValueError):
        barents.traj.to_orthogonal()

    print('converting to orthogonal')
    gr = barents.isel(trajectory=0).traj.to_orthogonal()
    assert gr.traj.is_orthogonal()


def test_trajectories_group_barents(barents):
    assert barents.traj.is_ragged()
    # print(barents)

    def a(t):
        print(t)
        return t.mean('obs')

    # Calculates mean of each trajectory
    p = barents.traj.trajectories().map(a)
    print(p)

def test_trajectories_group_omb(test_data):
    path_to_test_data = test_data / 'csv' / 'omb1.csv'
    ds = read_omb_csv(path_to_test_data)
    assert ds.traj.is_ragged()

    print(ds.lon.values)

    def a(t):
        print(t.lon.values)
        return t.dropna('obs').mean('obs')

    # Calculates mean of each trajectory
    p = ds.traj.trajectories().map(a)
    print(p)
