import numpy as np
import trajan as ta
import xarray as xr
import cf_xarray as _
import pytest
import pyproj
from pyproj import CRS, Transformer
from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from pytest import approx


def test_barents_detect_lonlat(barents):
    print(barents.traj.crs)

    crs = barents.traj.crs
    assert crs.to_cf()['grid_mapping_name'] == 'latitude_longitude'
    print(crs.to_cf())

    # get cartopy crs
    print(barents.traj.ccrs)


def test_barents_set_crs(barents):
    crs = barents.traj.crs
    barents = barents.traj.set_crs(crs)

    assert 'latitude_longitude' in barents
    assert len(barents.cf.grid_mapping_names) > 0
    assert barents.traj.crs == crs


def test_barents_tlat_tlon(barents):
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


# Proj 9.8.1 uses a more accurate ellipsoid for the 4326 (WGS84) (latlon)
# CRS. This no longer matches the PlateeCarre CRS in Cartopy. We need to make sure
# that GPS measurements and OpenDrift simulated positions are placed correctly in the
# CRS used by GPS, and that the coastline and map positions are placed in the
# correct positions.

def test_proj_4326_platecarree_default():
    # These used to be identical before Proj 9.8.1
    dcrs = CRS.from_proj4("+proj=lonlat +datum=WGS84 +ellps=WGS84 +no_defs")
    # gcrs = CRS.from_proj4(
    #     "+proj=eqc +ellps=WGS84 +lon_0=0.0 +to_meter=111319.4907932736 +vto_meter=1 +no_defs"
    # )
    gcrs = ccrs.PlateCarree()
    print(gcrs)

    t = Transformer.from_crs(dcrs, gcrs, always_xy=True)
    print(t.transform(5, 60))

    tlo, tla = t.transform(5, 60)
    assert tlo == approx(5., abs=0.00001)
    assert tla == approx(60., abs=0.00001)


def test_proj_4326_platecarree_forced_sphere():
    dcrs = CRS.from_proj4("+proj=lonlat +datum=WGS84 +ellps=WGS84 +no_defs")

    # Prior to Proj 9.8 this was the default, now it must be forced to match.
    WGS84_SEMIMAJOR_AXIS = 6378137
    gcrs = ccrs.PlateCarree(
        globe=ccrs.Globe(ellipse='WGS84',
                         semimajor_axis=WGS84_SEMIMAJOR_AXIS,
                         semiminor_axis=WGS84_SEMIMAJOR_AXIS))

    print(gcrs)
    # gcrs = CRS.from_proj4(
    #     "+proj=eqc +ellps=WGS84 +lon_0=0.0 +to_meter=111319.4907932736 +vto_meter=1 +no_defs"
    # )

    t = Transformer.from_crs(dcrs, gcrs, always_xy=True)
    print(t.transform(5, 60))

    tlo, tla = t.transform(5, 60)
    assert tlo == approx(5., abs=0.00001)
    assert tla == approx(60., abs=0.00001)


def test_proj_4326_geodetic_default():
    dcrs = CRS.from_proj4("+proj=lonlat +datum=WGS84 +ellps=WGS84 +no_defs")
    # gcrs = CRS.from_proj4(
    #     "+proj=eqc +ellps=WGS84 +lon_0=0.0 +to_meter=111319.4907932736 +vto_meter=1 +no_defs"
    # )
    gcrs = ccrs.Geodetic()
    print(gcrs)

    t = Transformer.from_crs(dcrs, gcrs, always_xy=True)
    print(t.transform(5, 60))

    tlo, tla = t.transform(5, 60)
    assert tlo == approx(5., abs=0.00001)
    assert tla == approx(60., abs=0.00001)


def test_proj_4326_geodetic_default_vs_forced_sphere():
    # dcrs = CRS.from_proj4("+proj=lonlat +datum=WGS84 +ellps=WGS84 +no_defs")
    dcrs = ccrs.Geodetic() # this is apparently always a sphere.

    WGS84_SEMIMAJOR_AXIS = 6378137
    gcrs = ccrs.Geodetic(
        globe=ccrs.Globe(ellipse='WGS84',
                         semimajor_axis=WGS84_SEMIMAJOR_AXIS,
                         semiminor_axis=WGS84_SEMIMAJOR_AXIS))

    print(gcrs)
    # gcrs = CRS.from_proj4(
    #     "+proj=eqc +ellps=WGS84 +lon_0=0.0 +to_meter=111319.4907932736 +vto_meter=1 +no_defs"
    # )

    t = Transformer.from_crs(dcrs, gcrs, always_xy=True)
    print(t.transform(5, 60))

    tlo, tla = t.transform(5, 60)
    assert tlo == approx(5., abs=0.00001)
    assert tla == approx(60., abs=0.00001)

def test_proj_platecarree_default_vs_forced_sphere():
    # This should fail on all versions of proj.
    dcrs = ccrs.PlateCarree() # in Proj 9.8 this uses an ellipsoid, before it is a sphere, identical to the below.

    WGS84_SEMIMAJOR_AXIS = 6378137
    gcrs = ccrs.PlateCarree(
        globe=ccrs.Globe(ellipse='WGS84',
                         semimajor_axis=WGS84_SEMIMAJOR_AXIS,
                         semiminor_axis=WGS84_SEMIMAJOR_AXIS))

    print(dcrs)
    print(gcrs)
    # gcrs = CRS.from_proj4(
    #     "+proj=eqc +ellps=WGS84 +lon_0=0.0 +to_meter=111319.4907932736 +vto_meter=1 +no_defs"
    # )

    t = Transformer.from_crs(dcrs, gcrs, always_xy=True)
    print(t.transform(5, 60))

    tlo, tla = t.transform(5, 60)
    assert tlo == approx(5., abs=0.00001)
    assert tla == approx(60., abs=0.00001)


def _random_points_in_extent(ds, n=100, seed=42):
    """Return n random (lon, lat) pairs uniformly distributed within the dataset extent."""
    lons = ds.lon.values
    lats = ds.lat.values
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    rng = np.random.default_rng(seed)
    rand_lons = rng.uniform(lon_min, lon_max, n)
    rand_lats = rng.uniform(lat_min, lat_max, n)
    return rand_lons, rand_lats


@pytest.mark.parametrize('with_random', [True, False], ids=['with_random', 'without_random'])
@pytest.mark.mpl_image_compare
def test_crs_bergen_and_opendrift(openoil, plot, with_random):
    """Plot Bergen and every tenth time-step of the opendrift trajectories, optionally with random points."""
    BERGEN_LON, BERGEN_LAT = 5.324, 60.389

    ds = openoil
    # proj = ds.traj.ccrs
    proj = ccrs.Mercator()

    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(8, 6))
    ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True)

    # Every tenth time-step of all trajectories
    lons = ds.lon.values[:, ::10].ravel()
    lats = ds.lat.values[:, ::10].ravel()
    ax.scatter(lons, lats, s=2, color='steelblue', alpha=0.5,
               transform=ccrs.Geodetic(), label='Trajectories (every 10th step)')

    if with_random:
        rand_lons, rand_lats = _random_points_in_extent(ds)
        ax.plot(rand_lons, rand_lats, 'g^', markersize=6,
                transform=ccrs.Geodetic(), label='Random points')

    # Bergen
    ax.plot(BERGEN_LON, BERGEN_LAT, 'r*', markersize=12,
            transform=ccrs.Geodetic(), label='Bergen')

    ax.legend(loc='upper left')
    ax.set_title('Bergen and OpenDrift trajectory positions')

    if plot:
        plt.show()

    return fig


@pytest.mark.parametrize('with_random', [True, False], ids=['with_random', 'without_random'])
@pytest.mark.mpl_image_compare
def test_crs_bergen_and_opendrift_traj(openoil, plot, with_random):
    """Plot Bergen and every tenth time-step using ds.traj.plot(), optionally with random points."""
    BERGEN_LON, BERGEN_LAT = 5.324, 60.389

    # Use trajan to set up the map and plot every 10th time-step
    _, ax = openoil.isel(time=slice(None, None, 10)).traj.plot()

    if with_random:
        rand_lons, rand_lats = _random_points_in_extent(openoil)
        ax.plot(rand_lons, rand_lats, 'g^', markersize=6,
                transform=ccrs.Geodetic(), label='Random points')

    # Bergen
    ax.plot(BERGEN_LON, BERGEN_LAT, 'r*', markersize=12,
            transform=ccrs.Geodetic(), label='Bergen')
    ax.legend(loc='upper left')

    if plot:
        plt.show()

    return ax.figure
