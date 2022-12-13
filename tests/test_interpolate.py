from numpy.testing import assert_almost_equal
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

def test_interpolate_barents(barents, plot):
    """Interpolate subset of drifter time series to 6-hourly means"""

    times = pd.date_range("2022-10-01", "2022-11-01", freq='6H')

    barents_gridded = barents.traj.gridtime(times)
    assert_almost_equal(barents_gridded.isel(trajectory=1).isel(time=100).lon, 19.94633948)

    if plot:
        drifter_names = barents['drifter_names'].data
        barents.traj.plot.set_up_map()  # Large enough to encompass both drifters
        barents.isel(trajectory=1).traj.plot(color='b', label=drifter_names[1])
        barents.isel(trajectory=0).traj.plot(color='r', label=drifter_names[0])

        barents_gridded.isel(trajectory=0).traj.plot(color='r', marker='x')
        barents_gridded.isel(trajectory=1).traj.plot(color='b', marker='x')

        plt.legend()
        plt.show()

def test_speed(barents, plot):
    """Implicitly also testing distance_to_next and time_to_next"""
    barents = barents.traj.insert_nan_where(barents.traj.time_to_next()>np.timedelta64(30, 'm'))
    speed = barents.traj.speed()
    speed = speed.where(speed<10).where(speed>0.01)
    
    assert_almost_equal(speed.max(), 1.287, 3)

    if plot:
        plt.hist(speed.values[~np.isnan(speed.values)], 100)
        plt.show()

def test_insert_nan_where(barents, plot):

    b2 = barents.traj.insert_nan_where(barents.traj.time_to_next()>np.timedelta64(30, 'm'))

    assert all([a == b for a, b in zip(barents.keys(), b2.keys())])
    assert b2.dims['obs'] == 3222

    if plot:
        barents.traj.plot(color='b', linewidth=2)
        b2.traj.plot(color='r')
        plt.show()

def test_drop_where(barents, plot):

    t2n = barents.traj.time_to_next()/np.timedelta64(1, 'm')
    assert_almost_equal(t2n.min(), 0.0166, 3)
    assert_almost_equal(barents.traj.speed().max(), 1008.3, 1) # Unreasonably large due to small timestep

    b2 = barents.traj.drop_where(t2n<5)  # Delete where timestep < 5 minutes
    t2n = b2.traj.time_to_next()/np.timedelta64(1, 'm')
    assert_almost_equal(t2n.min(), 24.166, 3)
    # After removing positions with very small time step, calculated maximum speed is reasonable
    assert_almost_equal(b2.traj.speed().max(), 1.287, 1)

    assert barents.dims['obs'] == 2287
    assert b2.dims['obs'] == 2287
    # Trimming off the empty positions at the end
    assert b2.dropna(dim='obs', how='all').dims['obs'] == 2279
