"""
Trajan API
"""
import collections
import numpy as np
import xarray as xr

from . import trajectory_accessor as _
from . import skill as _

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
        # Make sure that positions are sorted by time
        drifter_dict = collections.OrderedDict(sorted(drifter_dict.items()))

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
