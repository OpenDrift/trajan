import numpy as np
import trajan as ta
import xarray as xr
import cf_xarray as _
import matplotlib.pyplot as plt


def test_barents_detect_lonlat(barents, plot):
    print(barents.traj.crs)

    crs = barents.traj.crs
    assert crs.to_cf()['grid_mapping_name'] == 'latitude_longitude'
    print(crs.to_cf())

    # get cartopy crs
    print(barents.traj.ccrs)


def test_barents_set_crs(barents, plot):
    crs = barents.traj.crs
    barents = barents.traj.set_crs(crs)

    assert 'latitude_longitude' in barents
    assert len(barents.cf.grid_mapping_names) > 0
    assert barents.traj.crs == crs


def test_barents_tlat_tlon(barents, plot):
    np.testing.assert_array_equal(barents.lon, barents.traj.tlon)
    np.testing.assert_array_equal(barents.lat, barents.traj.tlat)

    crs = barents.traj.crs
    barents = barents.traj.set_crs(crs)

    np.testing.assert_array_equal(barents.lon, barents.traj.tlon)
    np.testing.assert_array_equal(barents.lat, barents.traj.tlat)


def test_barents_remove_crs(barents):
    assert 'lon' in barents
    barents = barents.traj.set_crs(None)
    print(barents.traj.crs)
    print(barents)

    assert 'lon' not in barents

    assert barents.traj.crs is None
