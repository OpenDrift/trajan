"""
Trajan API
"""
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from . import trajectory_accessor

# Sketch of an analysis package for trajectory datasets, which is independent of OpenDrift

def set_up_map(td=None, buffer=.1, corners=None, landmask='auto', lscale='auto', fast=True,
               ocean_color='white', land_color=cfeature.COLORS['land'], figsize=11):

    if corners is None:
        lonmin = td.lon.min() - buffer
        lonmax = td.lon.max() + buffer
        latmin = td.lat.min() - buffer
        latmax = td.lat.max() + buffer
    else:
        lonmin = corners[0]
        lonmax = corners[1]
        latmin = corners[2]
        latmax = corners[3]

    crs = ccrs.Mercator()
    globe = crs.globe
    gcrs = ccrs.PlateCarree(globe=crs.globe)  # For coordinate transform
    meanlat = (latmin + latmax) / 2
    aspect_ratio = float(latmax - latmin) / (float(lonmax - lonmin))
    aspect_ratio = aspect_ratio / np.cos(np.radians(meanlat))

    plt.close()  # Close any existing windows, so that they dont show up with new
    if aspect_ratio > 1:
        fig = plt.figure(figsize=(figsize / aspect_ratio, figsize))
    else:
        fig = plt.figure(figsize=(figsize, figsize * aspect_ratio))

    ax = fig.add_subplot(111, projection=crs)
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=gcrs)

    gl = ax.gridlines(gcrs, draw_labels=True)
    gl.top_labels = None

    fig.canvas.draw()
    fig.set_tight_layout(True)

    ####################
    # Landmask
    ####################
    if landmask == 'auto':
        # XXX: Opendrift cannot be imported in trajan.
        pass
        # from opendrift.readers import reader_global_landmask
        # reader_global_landmask.plot_land(ax, lonmin, latmin, lonmax, latmax,
        #                                  fast, ocean_color, land_color, lscale, globe)

    return ax, fig, gcrs

def plot(td, background=None, show=True, trajectory_kwargs={}, map_kwargs={}):

    # td: trajectory Xarray Dataset
    # background: DataArray with background field (not yet implemented)

    ax, fig, gcrs = set_up_map(td, **map_kwargs)

    if 'trajectory' in td.dims:
        num_trajectories = len(td.trajectory)
    else:
        num_trajectories = 1  # E.g. removed by averaging

    # Default trajectory options
    if 'alpha' not in trajectory_kwargs:
        # The more trajectories, the more transparent we make the lines
        min_alpha = 0.1
        max_trajectories = 5000.0
        alpha = min_alpha**(2 * (num_trajectories - 1) / (max_trajectories - 1))
        trajectory_kwargs['alpha'] = np.max((min_alpha, alpha))
    if 'color' not in trajectory_kwargs:
        trajectory_kwargs['color'] = 'gray'

    ####################
    # Plot trajectories
    ####################
    ax.plot(td.lon.T, td.lat.T, transform=gcrs, **trajectory_kwargs)

    if show is True:
        plt.show()
    else:
        return ax, fig, gcrs


def skillscore_liu_weissberg(lon_obs, lat_obs, lon_model, lat_model, tolerance_threshold=1):
    ''' calculate skill score from normalized cumulative seperation
    distance. Liu and Weisberg 2011.

    Returns the skill score bewteen 0. and 1.
    '''

    lon_obs = np.array(lon_obs)
    lat_obs = np.array(lat_obs)
    lon_model = np.array(lon_model)
    lat_model = np.array(lat_model)
    d = distance_between_trajectories(lon_obs, lat_obs, lon_model, lat_model)
    l = distance_along_trajectory(lon_obs, lat_obs)
    s = d.sum() / l.cumsum().sum()
    if tolerance_threshold==0:
        skillscore = 0
    else:
        skillscore = max(0, 1 - s/tolerance_threshold)

    return skillscore

def distance_between_trajectories(lon1, lat1, lon2, lat2):
    '''Calculate the distances [m] between two trajectories'''

    assert len(lon1) == len(lat1) == len(lat1) == len(lat2)
    geod = pyproj.Geod(ellps='WGS84')
    azimuth_forward, a2, distance = geod.inv(lon1, lat1, lon2, lat2)

    return distance

def distance_along_trajectory(lon, lat):
    '''Calculate the distances [m] between points along a trajectory'''

    geod = pyproj.Geod(ellps='WGS84')
    azimuth_forward, a2, distance = geod.inv(lon[1:], lat[1:], lon[0:-1], lat[0:-1])

    return distance

def trajectory_dict_to_dataset(trajectory_dict, attributes=None):
    """Create a CF-compatible trajectory file from dictionary of drifter positions

    Trajectory_dict shall have the following structure:
        {'buoy1_name': {
            time0: {'lon': lon0, 'lat': lat0},
            time1: {'lon': lon1, 'lat': lat1},
                ...
            timeN: {'lon': lonN, 'lat': latN}},
        {'buoy2_name': {
            ...
    """

    drifter_names = [td.dev for td in trajectory_dict]
    num_drifters = len(trajectory_dict)
    num_times = np.max([len(d) for dn, d in trajectory_dict.items()])
    # Allocate  arrays
    lon = np.empty((num_drifters, num_times))
    lon[:] = np.nan
    lat = np.empty((num_drifters, num_times))
    lat[:] = np.nan
    time = np.empty((num_drifters, num_times), dtype='datetime64[s]')
    time[:] = np.datetime64('nat')

    # Fill arrays with data from dictionaries
    for drifter_num, (drifter_name, drifter_dict) in enumerate(trajectory_dict.items()):
        t = slice(0, len(drifter_dict))
        lon[drifter_num, t] = np.array([di['lon'] for d, di in drifter_dict.items()])
        lat[drifter_num, t] = np.array([di['lat'] for d, di in drifter_dict.items()])
        time[drifter_num, t] = np.array(list(drifter_dict), dtype='datetime64[s]')

    # Remove empty attributes
    attributes = {a:v for a,v in attributes.items() if v != ''}

    # Create Xarray Dataset adhering to CF conventions h.4.1 for trajectory data
    ds = xr.Dataset(
        data_vars=dict(
            lon=(['trajectory', 'obs'], lon,
                {'standard_name': 'longitude', 'unit': 'degree_east'}),
            lat=(['trajectory', 'obs'], lat,
                {'standard_name': 'latitude', 'unit': 'degree_north'}),
            time=(['trajectory', 'obs'], time,
                {'standard_name': 'time'}),
            drifter_names=(['trajectory'], drifter_names,
                {'cf_role': 'trajectory_id', 'standard_name': 'platform_id'})
            ),
        attrs={'Conventions': 'CF-1.10',
               'featureType': 'trajectory',
               'geospatial_lat_min': np.nanmin(lat),
               'geospatial_lat_max': np.nanmax(lat),
               'geospatial_lon_min': np.nanmin(lon),
               'geospatial_lon_max': np.nanmax(lon),
               'time_coverage_start': str(np.nanmin(time)),
               'time_coverage_end': str(np.nanmax(time)),
                **attributes
               }
    )

    return ds
