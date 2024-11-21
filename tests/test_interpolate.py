from numpy.testing import assert_almost_equal
import numpy as np
import trajan as ta
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def test_interpolate_barents_hourly(barents):
    b = barents.traj.gridtime('1h')
    print(b)


def test_interpolate_barents(barents, plot):
    """Interpolate subset of drifter time series to 6-hourly means"""

    print(barents)
    times = pd.date_range("2022-10-01", "2022-11-01", freq='6h')

    barents_gridded = barents.traj.gridtime(times)
    assert_almost_equal(
        barents_gridded.isel(trajectory=1).isel(time=100).lon, 19.94633948)

    print(barents_gridded)

    if plot:
        trajectory = barents['trajectory'].data
        barents.traj.plot.set_up_map(
        )  # Large enough to encompass both drifters
        barents.isel(trajectory=1).traj.plot(color='b', label=trajectory[1])
        barents.isel(trajectory=0).traj.plot(color='r', label=trajectory[0])

        barents_gridded.isel(trajectory=0).traj.plot(color='r', marker='x')
        barents_gridded.isel(trajectory=1).traj.plot(color='b', marker='x')

        plt.legend()
        plt.show()


def test_distance(barents):
    barents = barents.traj.gridtime('1h')
    b0 = barents.isel(trajectory=0)
    b1 = barents.isel(trajectory=1)

    d = b0.traj.distance_to(b1)
    print(d)


def test_length(barents):
    l = barents.traj.length()
    print(l)

    lg = barents.traj.gridtime('1H').traj.length()
    print(lg)

    np.testing.assert_allclose(l, lg, atol=20000)

    bb = barents.isel(trajectory=0).traj.iseltime([0, -1]).traj.to_1d()
    np.testing.assert_almost_equal(bb.traj.distance_to_next().isel(time=0),
                                   bb.traj.length())

    np.testing.assert_almost_equal(l, [394484.37231936, 1990989.5283971])


def test_distance_single_point(barents):
    barents = barents.traj.gridtime('1h')
    ds = barents.isel(time=4).isel(trajectory=0)
    print("single:", ds)

    barents = barents.dropna(dim='time', how='all')

    d = barents.traj.distance_to(ds)
    print(d)


def test_interpolate_1d_barents(barents):
    times = pd.date_range("2022-10-01", "2022-11-01", freq='6h')

    barents_gridded = barents.traj.gridtime(times)

    # Try gridding the gridded
    b2 = barents_gridded.traj.gridtime(times)

    np.testing.assert_array_equal(barents_gridded.time, b2.time)

    # Try gridding the gridded
    b3 = barents_gridded.traj.gridtime(b2.time)

    np.testing.assert_array_equal(barents_gridded.time, b3.time)

    times = pd.date_range("2022-10-12", "2022-11-01", freq='6h')
    b4 = barents_gridded.traj.gridtime(times)

    np.testing.assert_array_equal(times, b4.time)


def test_interpolate_non_floats(drifter_csv):
    dc = ta.read_csv(drifter_csv,
                     name='Device',
                     time='Time',
                     lon='Longitude',
                     lat='Latitude')
    dcg = dc.traj.gridtime('1h')

    assert 'time' in dcg.sizes
    assert 'trajectory' in dcg.sizes

    print(dc, dcg)


def test_interpolate_barents_between_trajs(barents):
    barents = barents.traj.gridtime('1h')
    b0 = barents.isel(trajectory=0)
    b1 = barents.isel(trajectory=1)

    b0 = b0.traj.gridtime(b1.time)
    np.testing.assert_array_equal(b0.time, b1.time)


def test_speed(barents, plot):
    """Implicitly also testing distance_to_next and time_to_next"""
    barents = barents.traj.insert_nan_where(
        barents.traj.time_to_next() > np.timedelta64(60, 'm'))
    barents = barents.traj.drop_where(
        barents.traj.time_to_next()
        < np.timedelta64(5, 'm'))  # Delete where timestep < 5 minutes
    speed = barents.traj.speed()
    speed = speed.where(speed > 0.01)  # Mask where drifter is on land

    assert_almost_equal(speed.max(), 1.287, 3)
    assert_almost_equal(speed.mean(), 0.461, 3)

    # Gridding to hourly
    bh = barents.traj.gridtime('1h')
    speed_bh = bh.traj.speed()
    speed_bh = speed_bh.where(speed_bh > 0.01)

    assert_almost_equal(speed_bh.max(), 1.261,
                        3)  # Slightly different after gridding
    assert_almost_equal(speed_bh.mean(), 0.459, 3)

    if plot:
        plt.hist(speed.values[~np.isnan(speed.values)],
                 100,
                 color='r',
                 label='Original')
        plt.hist(speed_bh.values[~np.isnan(speed_bh.values)],
                 100,
                 color='b',
                 label='Hourly gridded')
        plt.legend()
        plt.xlabel('Drifter speed  [m/s]')
        plt.ylabel('Number')
        plt.show()

def test_speed_2d(barents):
    s = barents.traj.speed()
    print(s)

def test_insert_nan_where(barents, plot):

    b2 = barents.traj.insert_nan_where(
        barents.traj.time_to_next() > np.timedelta64(30, 'm'))

    assert all([a == b for a, b in zip(barents.keys(), b2.keys())])
    assert b2.sizes['obs'] == 3222

    if plot:
        barents.traj.plot(color='b', linewidth=2)
        b2.traj.plot(color='r')
        plt.show()


def test_drop_where(barents, plot):

    t2n = barents.traj.time_to_next() / np.timedelta64(1, 'm')
    assert_almost_equal(t2n.min(), 0.0166, 3)
    assert_almost_equal(barents.traj.speed().max(), 1008.3,
                        1)  # Unreasonably large due to small timestep

    b2 = barents.traj.drop_where(t2n < 5)  # Delete where timestep < 5 minutes
    t2n = b2.traj.time_to_next() / np.timedelta64(1, 'm')
    assert_almost_equal(t2n.min(), 24.166, 3)
    # After removing positions with very small time step, calculated maximum speed is reasonable
    assert_almost_equal(b2.traj.speed().max(), 1.287, 1)

    assert barents.sizes['obs'] == 2287
    assert b2.sizes['obs'] == 2285
    # Trimming off the empty positions at the end
    assert b2.dropna(dim='obs', how='all').sizes['obs'] == 2279
