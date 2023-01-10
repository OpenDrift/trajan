"""
Trajan API
"""
import collections
import numpy as np
import xarray as xr

from . import trajectory_accessor as _
from . import skill as _

def trajectory_dict_to_dataset(trajectory_dict, variable_attributes=None, global_attributes=None):
    """Create a CF-compatible trajectory file from dictionary of drifter positions

    Trajectory_dict shall have the following structure:
        {'buoy1_name': {
            time0: {'lon': lon0, 'lat': lat0, 'variable1': var1_0, ... 'variableM': varM_0},
            time1: {'lon': lon1, 'lat': lat1, 'variable1': var1_1, ... 'variableM': varM_1},
                ...
            timeN: {'lon': lonN, 'lat': latN, 'variable1': var1_N, ... 'variableM': varM_N}},
        {'buoy2_name': {
            ...
    """

    if variable_attributes is None:
        variable_attributes = dict()

    if global_attributes is None:
        global_attributes = dict()

    drifter_names = list(trajectory_dict)
    num_drifters = len(drifter_names)
    num_times = np.max([len(d) for dn, d in trajectory_dict.items()])
    # Allocate  arrays
    variables = []
    for dn, d in trajectory_dict.items():
        v = list(d[list(d)[0]])
        for var in v:
            if var not in variables:
                variables.extend([var])
    datavars = {}
    for var in variables:
        datavars[var] = np.empty((num_drifters, num_times))
        datavars[var][:] = np.nan
        if var not in variable_attributes:
            if var == 'lat':
                variable_attributes[var] = {'standard_name': 'latitude', 'unit': 'degree_north'}
            elif var == 'lon':
                variable_attributes[var] = {'standard_name': 'longitude', 'unit': 'degree_east'}
            else:
                variable_attributes[var] = {}
    time = np.empty((num_drifters, num_times), dtype='datetime64[s]')
    time[:] = np.datetime64('nat')

    # Fill arrays with data from dictionaries
    for drifter_num, (drifter_name, drifter_dict) in enumerate(trajectory_dict.items()):
        # Make sure that positions are sorted by time
        drifter_dict = collections.OrderedDict(sorted(drifter_dict.items()))
        t = slice(0, len(drifter_dict))
        for var in variables:
            datavars[var][drifter_num, t] = np.array([di[var] for d, di in drifter_dict.items()])
        time[drifter_num, t] = np.array(list(drifter_dict), dtype='datetime64[s]')

    datavars = {v: (['trajectory', 'obs'], datavars[v], variable_attributes[v]) for v in variables}
    datavars['time'] = (['trajectory', 'obs'], time, {'standard_name': 'time'})
    datavars['drifter_names'] = (['trajectory'], drifter_names,
                {'cf_role': 'trajectory_id', 'standard_name': 'platform_id'})

    # Remove empty attributes
    global_attributes = {a:v for a,v in global_attributes.items() if v != ''}

    # Create Xarray Dataset adhering to CF conventions h.4.1 for trajectory data
    ds = xr.Dataset(
        data_vars=datavars,
        attrs={'Conventions': 'CF-1.10',
               'featureType': 'trajectory',
               'geospatial_lat_min': np.nanmin(datavars['lat'][1]),
               'geospatial_lat_max': np.nanmax(datavars['lat'][1]),
               'geospatial_lon_min': np.nanmin(datavars['lon'][1]),
               'geospatial_lon_max': np.nanmax(datavars['lon'][1]),
               'time_coverage_start': str(np.nanmin(datavars['time'][1])),
               'time_coverage_end': str(np.nanmax(datavars['time'][1])),
                **global_attributes
               }
    )

    return ds
