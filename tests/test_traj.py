import pytest
import numpy as np
import xarray as xr
import pyproj
from datetime import datetime, timedelta
from trajan.traj import Traj, grid_area


@pytest.fixture
def mock_dataset():
    # Making a 1D dataset for basic testing
    times = np.array([datetime(2023, 1, 1) + timedelta(hours=i) for i in range(5)])
    #times = np.repeat(times[np.newaxis, :], 2, axis=0)
    data = {
        "lon": (["trajectory", "obs"], np.random.rand(2, 5)),
        "lat": (["trajectory", "obs"], np.random.rand(2, 5)),
        "time": (["obs"], times)
        #"time": (["trajectory", "obs"], times)
    }
    coords = {"trajectory": [0, 1], "obs": range(5)}
    return xr.Dataset(data, coords=coords)


@pytest.fixture
def mock_traj(mock_dataset):
    return Traj(mock_dataset, trajectory_dim="trajectory", obs_dim="obs", time_varname="time")


def test_grid_area():
    lons = np.linspace(0, 10, 5)
    lats = np.linspace(0, 10, 5)
    area = grid_area(lons, lats)
    assert isinstance(area, xr.DataArray)
    assert area.name == "grid_area"
    #assert "lat" in area.dims
    #assert "lon" in area.dims


def test_index_of_last(mock_traj):
    result = mock_traj.index_of_last()
    assert isinstance(result, xr.DataArray)
    assert result.name == "index_of_last"
    assert "trajectory" in result.dims


#def test_speed(mock_traj):
#    result = mock_traj.speed()
#    assert isinstance(result, xr.DataArray)
#    assert result.name == "speed"
#    assert "obs" in result.dims


#def test_time_to_next(mock_traj):
#    result = mock_traj.time_to_next()
#    assert isinstance(result, xr.DataArray)
#    assert result.name == "time_to_next"
#    assert "obs" in result.dims


def test_distance_to_next(mock_traj):
    result = mock_traj.distance_to_next()
    assert isinstance(result, xr.DataArray)
    assert result.name is None
    assert "obs" in result.dims


def test_azimuth_to_next(mock_traj):
    result = mock_traj.azimuth_to_next()
    assert isinstance(result, xr.DataArray)
    assert result.name is None
    assert "obs" in result.dims


#def test_velocity_components(mock_traj):
#    u, v = mock_traj.velocity_components()
#    assert isinstance(u, xr.DataArray)
#    assert isinstance(v, xr.DataArray)
#    assert u.name == "u_velocity"
#    assert v.name == "v_velocity"
#    assert "obs" in u.dims
#    assert "obs" in v.dims


def test_get_area_convex_hull(mock_traj):
    result = mock_traj.get_area_convex_hull()
    assert isinstance(result, xr.DataArray)
    assert result.name == "convex_hull_area"
    assert result.attrs["units"] == "m2"


def test_make_grid(mock_traj):
    result = mock_traj.make_grid(dx=10000, dy=None)
    assert isinstance(result, xr.Dataset)
    assert "cell_area" in result.data_vars


def test_crop(mock_traj):
    result = mock_traj.crop(lonmin=0, lonmax=5, latmin=0, latmax=5)
    assert isinstance(result, xr.Dataset)


def test_contained_in(mock_traj):
    result = mock_traj.contained_in(lonmin=0, lonmax=5, latmin=0, latmax=5)
    assert isinstance(result, xr.Dataset)


def test_assign_cf_attrs(mock_traj):
    result = mock_traj.assign_cf_attrs(creator_name="Test", creator_email="test@example.com")
    assert isinstance(result, xr.Dataset)
    assert result.attrs["creator_name"] == "Test"
    assert result.attrs["creator_email"] == "test@example.com"