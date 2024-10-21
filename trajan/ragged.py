from .traj2d import Traj2d

import numpy as np

import logging

logger = logging.getLogger(__name__)


class ContiguousRagged(Traj2d):
    def __init__(self, ds, obsdim, timedim, rowvar):
        ds_converted_to_traj2d = _convert_to_Traj2d(ds, obsdim, timedim, rowvar)
        super().__init__(ds_converted_to_traj2d, "obs", "time")


    def _convert_to_Traj2d(ds, obsdim, timedim, rowvar):
        nbr_trajectories = len(ds.obsdim)

        # find the longest trajectory
        longest_trajectory = np.max(ds.rowvar.to_numpy())

        # find the array of instruments
        array_instruments = ds.obsdim.to_numpy()

        # generate the empty array of time, lat, lon per instrument
        array_time = np.full(
            (nbr_trajectories, longest_trajectory),
            np.datetime64('nat'),
            dtype='datetime64[ns]'
        )

        array_lat = np.full((nbr_trajectories, longest_trajectory), np.nan)
        array_lon = np.full((nbr_trajectories, longest_trajectory), np.nan)

        # fill with the data
        start_index = 0
        for crrt_index, crrt_rowsize in enumerate(ds.rowvar.to_numpy()):
            end_index = start_index + crrt_rowsize

            array_time[crrt_index, :crrt_rowsize] = ds.time[start_index:end_index]
            array_lat[crrt_index, :crrt_rowsize] = ds.latitude[start_index:end_index]
            array_lon[crrt_index, :crrt_rowsize] = ds.longitude[start_index:end_index]

            start_index = end_index
    
        # generate the xarray in trajan format
        # TODO: convert key variables by hand; other variables: convert all of them automatically
        ds_converted_to_traj2d = xr.Dataset(
            {
                # meta vars
                'trajectory':
                xr.DataArray(data=array_instruments,
                             dims=['trajectory'],
                             attrs={
                                 "cf_role": "trajectory_id",
                                 "standard_name": "platform_id",
                                 "units": "1",
                                 "long_name": "ID / name of each buoy present in the deployment data.",
                             }).astype(str),

                # trajectory vars
                'time':
                    xr.DataArray(dims=["trajectory", "obs"],
                                 data=array_time,
                                 attrs={
                                     "standard_name": "time",
                                     "long_name": "Time for the GNSS position records.",
                    }),
                #
                'lat':
                xr.DataArray(dims=["trajectory", "obs"],
                             data=array_lat,
                             attrs={
                                 "_FillValue": "NaN",
                                 "standard_name": "latitude",
                                 "units": "degree_north",
                }),
                #
                'lon':
                xr.DataArray(dims=["trajectory", "obs"],
                             data=array_lon,
                             attrs={
                                 "_FillValue": "NaN",
                                 "standard_name": "longitude",
                                 "units": "degree_east",
                }),
            }
        )

        # %%
        # TODO: perfect forwarding of the other attributes?

        return ds_converted_to_traj2d
