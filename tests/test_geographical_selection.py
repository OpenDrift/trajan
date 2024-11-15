import trajan as ta
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_clip(barents):

    # Clipping dataset to:
    lonmin = 24
    lonmax = 28
    latmin = 76.5
    latmax = 77

    # Check that original coverage is larger than bounding box
    assert barents.traj.tlon.min(skipna=True) < lonmin
    assert barents.traj.tlon.max(skipna=True) > lonmax
    assert barents.traj.tlat.min(skipna=True) < latmin
    assert barents.traj.tlat.max(skipna=True) > latmax

    bc = barents.traj.crop(lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)

    # Check that cropped coverage is within bounding box
    assert bc.traj.tlon.min(skipna=True) > lonmin
    assert bc.traj.tlon.max(skipna=True) < lonmax
    assert bc.traj.tlat.min(skipna=True) > latmin
    assert bc.traj.tlat.max(skipna=True) < latmax

def test_contained_in(barents):

    # Check that both trajectories are within a large area
    bc = barents.traj.contained_in(lonmin=0, lonmax=100, latmin=50, latmax=80)
    assert bc.sizes[bc.traj.trajectory_dim] == 2

    # Check that only one of the trajectories are fully within given area
    bc = barents.traj.contained_in(lonmin=15, lonmax=28, latmin=72, latmax=77.2)
    assert bc.sizes[bc.traj.trajectory_dim] == 1

    # Check that none of the trajectories are fully within a small area (although both intersects)
    bc = barents.traj.contained_in(lonmin=22, lonmax=27, latmin=76.5, latmax=77.2)
    assert bc.sizes[bc.traj.trajectory_dim] == 0
