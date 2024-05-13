"""
Trajan API
"""
import collections
import numpy as np
import xarray as xr
import pandas as pd

from . import trajectory_accessor as _
from . import skill as _
from . import readers as _


def read_csv(f, **kwargs):
    """
    Construct a CF-compliant trajectory dataset from a CSV file. A thin wrapper around: :meth:`from_dataframe`.
    """
    df = pd.read_csv(f)
    return from_dataframe(df, **kwargs)


def from_dataframe(df: pd.DataFrame,
                   lon='lon',
                   lat='lat',
                   time='time',
                   name=None,
                   *,
                   __test_condense__=False):
    """
    Construct a CF-compliant trajectory dataset from a `pd.DataFrame` of positions.

    Args:

        `lon`: name of column containing longitudes.

        `lat`: name of column containing latitudes.

        `time`: name of column containing timestamps (parsable by pandas).

        `name`: name of column to be used for drifter names.

    Returns:

        `ds`: a CF-compliant trajectory `xarray.Dataset`.


    Constructing a dataset from arrays of postions and time:
    --------------------------------------------------------

    .. testcode::

        import pandas as pd
        import trajan as ta

        # Construct some synthetic data
        lon = np.linspace(5, 10, 50)
        lat = np.linspace(60, 70, 50)
        temp = np.linspace(10, 15, 50)
        time = pd.date_range('2023-01-01', '2023-01-14', periods=50)

        # Construct a pandas.DataFrame
        ds = pd.DataFrame({'lon': lon, 'lat': lat, 'time': time, 'temp': temp, 'name': 'My drifter'})
        ds = ta.from_dataframe(ds, name='name')
        print(ds)

    .. testoutput::

        <xarray.Dataset>
        Dimensions:     (trajectory: 1, obs: 50)
        Coordinates:
          * trajectory  (trajectory) <U10 'My drifter'
          * obs         (obs) int64 0 1 2 3 4 5 6 7 8 9 ... 41 42 43 44 45 46 47 48 49
        Data variables:
            lon         (trajectory, obs) float64 5.0 5.102 5.204 ... 9.796 9.898 10.0
            lat         (trajectory, obs) float64 60.0 60.2 60.41 ... 69.59 69.8 70.0
            time        (trajectory, obs) datetime64[ns] 2023-01-01 ... 2023-01-14
            temp        (trajectory, obs) float64 10.0 10.1 10.2 ... 14.8 14.9 15.0
        Attributes:
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   60.0
            geospatial_lat_max:   70.0
            geospatial_lon_min:   5.0
            geospatial_lon_max:   10.0
            time_coverage_start:  2023-01-01T00:00:00
            time_coverage_end:    2023-01-14T00:00:00

    Often you might want to add some attributes:

    .. testcode::

        ds = ds.assign_attrs({'Author': 'Albus Dumbledore'})
        print(ds)

    .. testoutput::

        <xarray.Dataset>
        Dimensions:     (trajectory: 1, obs: 50)
        Coordinates:
          * trajectory  (trajectory) <U10 'My drifter'
          * obs         (obs) int64 0 1 2 3 4 5 6 7 8 9 ... 41 42 43 44 45 46 47 48 49
        Data variables:
            lon         (trajectory, obs) float64 5.0 5.102 5.204 ... 9.796 9.898 10.0
            lat         (trajectory, obs) float64 60.0 60.2 60.41 ... 69.59 69.8 70.0
            time        (trajectory, obs) datetime64[ns] 2023-01-01 ... 2023-01-14
            temp        (trajectory, obs) float64 10.0 10.1 10.2 ... 14.8 14.9 15.0
        Attributes:
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   60.0
            geospatial_lat_max:   70.0
            geospatial_lon_min:   5.0
            geospatial_lon_max:   10.0
            time_coverage_start:  2023-01-01T00:00:00
            time_coverage_end:    2023-01-14T00:00:00
            Author:               Albus Dumbledore

    """
    df = df.copy()
    df.index.names = ['obs']
    df[time] = pd.to_datetime(df[time], format='mixed')

    df = df.rename(columns={lat: 'lat', lon: 'lon', time: 'time'})

    if name is not None:
        df = df.rename(columns={name: 'trajectory'})
    else:
        df['trajectory'] = 'Drifter 1'

    # Convert to UTC and remove tz-info. Xarray does not support tz-aware
    # datetimes.
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_convert(None)

    # Classify trajectories based on drifter names.
    df = df.set_index(['trajectory', df.index])
    df = df.to_xarray()

    if not __test_condense__:
        df = df.traj.condense_obs()

    df = df.traj.assign_cf_attrs()

    return df


def trajectory_dict_to_dataset(trajectory_dict,
                               variable_attributes=None,
                               global_attributes=None):
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
                variable_attributes[var] = {
                    'standard_name': 'latitude',
                    'unit': 'degree_north'
                }
            elif var == 'lon':
                variable_attributes[var] = {
                    'standard_name': 'longitude',
                    'unit': 'degree_east'
                }
            else:
                variable_attributes[var] = {}
    time = np.empty((num_drifters, num_times), dtype='datetime64[s]')
    time[:] = np.datetime64('nat')

    # Fill arrays with data from dictionaries
    for drifter_num, (drifter_name,
                      drifter_dict) in enumerate(trajectory_dict.items()):
        # Make sure that positions are sorted by time
        drifter_dict = collections.OrderedDict(sorted(drifter_dict.items()))
        t = slice(0, len(drifter_dict))
        for var in variables:
            datavars[var][drifter_num, t] = np.array(
                [di[var] for d, di in drifter_dict.items()])
        time[drifter_num, t] = np.array(list(drifter_dict),
                                        dtype='datetime64[s]')

    datavars = {
        v: (['trajectory', 'obs'], datavars[v], variable_attributes[v])
        for v in variables
    }
    datavars['time'] = (['trajectory', 'obs'], time, {'standard_name': 'time'})
    datavars['drifter_names'] = (['trajectory'], drifter_names, {
        'cf_role': 'trajectory_id',
        'standard_name': 'platform_id'
    })

    # Remove empty attributes
    global_attributes = {a: v for a, v in global_attributes.items() if v != ''}

    # Create Xarray Dataset adhering to CF conventions h.4.1 for trajectory data
    ds = xr.Dataset(data_vars=datavars,
                    attrs={
                        'Conventions': 'CF-1.10',
                        'featureType': 'trajectory',
                        'geospatial_lat_min': np.nanmin(datavars['lat'][1]),
                        'geospatial_lat_max': np.nanmax(datavars['lat'][1]),
                        'geospatial_lon_min': np.nanmin(datavars['lon'][1]),
                        'geospatial_lon_max': np.nanmax(datavars['lon'][1]),
                        'time_coverage_start':
                        str(np.nanmin(datavars['time'][1])),
                        'time_coverage_end':
                        str(np.nanmax(datavars['time'][1])),
                        **global_attributes
                    })

    return ds

