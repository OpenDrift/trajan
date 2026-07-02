import numpy as np
import trajan as ta
import xarray as xr
import cf_xarray as _
import pytest
import pyproj
from pyproj import CRS, Transformer
from cartopy import crs as ccrs
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
def proj_version():
    return [int(vv) for vv in pyproj.__proj_version__.split('.')]


def proj_gt_98():
    v = proj_version()
    return v[0] >= 9 and v[1] >= 8


@pytest.mark.xfail(
    condition=proj_gt_98(),
    reason=
    'EPSG:4326 and cartopy PlateCarree are no longer identical in proj>=9.8',
    strict=True)
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


@pytest.mark.xfail(
    condition=not proj_gt_98(),
    reason='EPSG:4326 and cartopy PlateCarree are identical in proj>=9.8',
    strict=True)
def test_proj_4326_platecarree_ellipsoid():
    # These used to be identical before Proj 9.8.1
    dcrs = CRS.from_proj4("+proj=lonlat +datum=WGS84 +ellps=WGS84 +no_defs")

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


def test_proj_4326_geodetic_default_vs_ellipsoid():
    # dcrs = CRS.from_proj4("+proj=lonlat +datum=WGS84 +ellps=WGS84 +no_defs")
    dcrs = ccrs.Geodetic()

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

@pytest.mark.xfail(
    reason='cartopy PlateCarree with and without ellipsoid should never be equal.',
    strict=True)
def test_proj_platecarree_default_vs_ellipsoid():
    # This should fail on all versions of proj.
    dcrs = ccrs.PlateCarree()

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
