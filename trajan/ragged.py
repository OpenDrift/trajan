from .traj import Traj
from .plot import Plot

import numpy as np
import xarray as xr

import logging

logger = logging.getLogger(__name__)

class ContiguousRagged(Traj):
    """An unstructured dataset, where each trajectory may have observations at different times, and all the data for the different trajectories are stored in single arrays with one dimension, contiguously, one trajectory after the other. Typically from a collection of drifters. This class convert continous ragged datasets into 2d datasets, so that the Traj2d methods can be leveraged."""

    rowvar: str  # TODO: Should have a more precise name than rowvar

    def __init__(self, ds, trajectory_dim, obs_dim, time_varname, rowsizevar):
        self.rowvar = rowsizevar
        super().__init__(ds, trajectory_dim, obs_dim, time_varname)

    def to_2d(self, obs_dim='obs'):
        """This actually converts a contiguous ragged xarray Dataset into an xarray Dataset that follows the Traj2d conventions."""
        global_attrs = self.ds.attrs

        nbr_trajectories = len(self.ds[self.trajectory_dim])

        # find the longest trajectory
        longest_trajectory = np.max(self.ds[self.rowvar].to_numpy())

        # generate the xarray in trajan format

        # the trajectory dimension special case (as it is a different kind, and has a different dim than other variables)

        array_instruments = self.ds[self.trajectory_dim].to_numpy()

        # the time var (special case as it is of a different type)

        array_time = np.full((nbr_trajectories, longest_trajectory),
                             np.datetime64('nat'),
                             dtype='datetime64[ns]')

        start_index = 0
        for crrt_index, crrt_rowsize in enumerate(
                self.ds[self.rowvar].to_numpy()):
            end_index = start_index + crrt_rowsize
            array_time[crrt_index, :crrt_rowsize] = self.ds[
                self.time_varname][start_index:end_index]
            start_index = end_index

        # it seems that we need to build the "backbone" of the Dataset independently first
        # (I have tried to put everything in a dict spec and build the Dataset in one go as it felt more elegant, but it did not work)

        ds_converted_to_traj2d = xr.Dataset({
            # meta vars
            'trajectory':
            xr.DataArray(
                data=array_instruments,
                dims=['trajectory'],
                attrs={
                    "cf_role":
                    "trajectory_id",
                    "standard_name":
                    "platform_id",
                    "units":
                    "1",
                    "long_name":
                    "ID / name of each buoy present in the deployment data.",
                }).astype(str),

            # trajectory vars
            'time':
            xr.DataArray(dims=["trajectory", obs_dim],
                         data=array_time,
                         attrs={
                             "standard_name": "time",
                             "long_name":
                             "Time for the GNSS position records.",
                         }),
        })

        # now add all "normal" variables
        # NOTE: for now, we only consider scalar vars; if we want to consider more complex vars (e.g., spectra), this will need updated
        # NOTE: such an update would typically need to look at the dims of the variable, and if there are additional dims to obs_dim, create a higer dim variable

        for crrt_data_var in self.ds.data_vars:
            attrs = self.ds[crrt_data_var].attrs

            if crrt_data_var == self.rowvar:
                continue

            if len(self.ds[crrt_data_var].dims
                   ) != 1 or self.ds[crrt_data_var].dims[0] != self.obs_dim:
                raise ValueError(
                    f"data_vars element {crrt_data_var} has dims {self.ds[crrt_data_var].dims}, expected {(self.obs_dim,)}"
                )

            crrt_var = np.full((nbr_trajectories, longest_trajectory), np.nan)

            start_index = 0

            for crrt_index, crrt_rowsize in enumerate(
                    self.ds[self.rowvar].to_numpy()):
                end_index = start_index + crrt_rowsize

                crrt_var[crrt_index, :crrt_rowsize] = self.ds[crrt_data_var][
                    start_index:end_index]

                start_index = end_index

            # somehow, the renaming to lat and lon is not called; need to do this by hand for now
            # NOTE: maybe there is a better way to call the renaming to lat and lon which is inherited from some other trajan function or class, but if so I am not sure how
            # NOTE: for now, not renaming here was creating a crash for example when plotting (no attribute lon)

            if crrt_data_var == "longitude":
                crrt_data_var = "lon"

            if crrt_data_var == "latitude":
                crrt_data_var = "lat"

            ds_converted_to_traj2d[crrt_data_var] = \
                xr.DataArray(dims=["trajectory", obs_dim],
                             data=crrt_var,
                             attrs=attrs)

        # copy initial global attributes
        ds_converted_to_traj2d = ds_converted_to_traj2d.assign_attrs(
            global_attrs)
        ds_converted_to_traj2d = ds_converted_to_traj2d.assign_attrs(
            trajan_modified=
            "this was initially a contiguous ragged Dataset, which was converted to a Traj2d dataset by trajan"
        )

        return ds_converted_to_traj2d

    @property
    def plot(self) -> Plot:
        return self.to_2d().traj.plot

    def timestep(self, average=np.median):
        return self.to_2d().traj.timestep(average)

    def gridtime(self, times, time_varname=None, round=True):
        return self.to_2d().traj.gridtime(times, time_varname, round)

