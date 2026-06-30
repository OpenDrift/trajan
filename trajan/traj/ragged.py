import xarray as xr
import numpy as np
import pandas as pd
import logging

from ..traj import Traj, ensure_time_dim, inherit_docstrings

logger = logging.getLogger(__name__)


def __require_obs_dim__(f):
    """This decorator is for methods of Traj that require a time or obs dimension to work."""

    def wrapper(*args, **kwargs):
        if args[0].obs_dim is None:
            raise ValueError(f'{f} requires an obs or time dimension')
        return f(*args, **kwargs)

    return wrapper


@inherit_docstrings
class TrajRagged(Traj):
    """
    A unstructured dataset, where each trajectory may have observations at different times. Typically from a collection of drifters.
    """

    def __init__(self, ds, trajectory_dim, obs_dim, time_varname):
        super().__init__(ds, trajectory_dim, obs_dim, time_varname)

    @staticmethod
    def from_contiguous(ds, trajectory_dim, obs_dim, time_varname,
                        rowvar) -> Traj:
        """
        An unstructured dataset, where each trajectory may have observations at different times, and all the data for the different trajectories are stored in single arrays with one dimension, contiguously, one trajectory after the other. Typically from a collection of drifters.

        This method converts ContinousRagged datasets into Ragged datasets.
        """
        global_attrs = ds.attrs

        nbr_trajectories = len(ds[trajectory_dim])

        # find the longest trajectory
        longest_trajectory = np.max(ds[rowvar].to_numpy())

        # generate the xarray in trajan format

        # the trajectory dimension special case (as it is a different kind, and has a different dim than other variables)

        array_instruments = ds[trajectory_dim].to_numpy()

        # the time var (special case as it is of a different type)

        array_time = np.full((nbr_trajectories, longest_trajectory),
                             np.datetime64('nat'),
                             dtype='datetime64[ns]')

        start_index = 0
        for crrt_index, crrt_rowsize in enumerate(ds[rowvar].to_numpy()):
            end_index = start_index + crrt_rowsize
            array_time[crrt_index, :crrt_rowsize] = ds[time_varname][
                start_index:end_index]
            start_index = end_index

        # it seems that we need to build the "backbone" of the Dataset independently first
        # (I have tried to put everything in a dict spec and build the Dataset in one go as it felt more elegant, but it did not work)

        ds_converted_to_trajRagged = xr.Dataset({
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

        for crrt_data_var in ds.data_vars:
            attrs = ds[crrt_data_var].attrs

            if crrt_data_var == rowvar:
                continue

            if len(ds[crrt_data_var].dims
                   ) != 1 or ds[crrt_data_var].dims[0] != obs_dim:
                raise ValueError(
                    f"data_vars element {crrt_data_var} has dims {ds[crrt_data_var].dims}, expected {(obs_dim,)}"
                )

            crrt_var = np.full((nbr_trajectories, longest_trajectory), np.nan)

            start_index = 0

            for crrt_index, crrt_rowsize in enumerate(ds[rowvar].to_numpy()):
                end_index = start_index + crrt_rowsize

                crrt_var[crrt_index, :crrt_rowsize] = ds[crrt_data_var][
                    start_index:end_index]

                start_index = end_index

            # somehow, the renaming to lat and lon is not called; need to do this by hand for now
            # NOTE: maybe there is a better way to call the renaming to lat and lon which is inherited from some other trajan function or class, but if so I am not sure how
            # NOTE: for now, not renaming here was creating a crash for example when plotting (no attribute lon)

            if crrt_data_var == "longitude":
                crrt_data_var = "lon"

            if crrt_data_var == "latitude":
                crrt_data_var = "lat"

            ds_converted_to_trajRagged[crrt_data_var] = \
                xr.DataArray(dims=["trajectory", obs_dim],
                             data=crrt_var,
                             attrs=attrs)

        # copy initial global attributes
        ds_converted_to_trajRagged = ds_converted_to_trajRagged.assign_attrs(
            global_attrs)
        ds_converted_to_trajRagged = ds_converted_to_trajRagged.assign_attrs(
            trajan_modified=
            "this was initially a contiguous ragged Dataset, which was converted to a TrajRagged dataset by trajan"
        )

        return TrajRagged(ds_converted_to_trajRagged, trajectory_dim, obs_dim,
                          time_varname)

    def timestep(self, average=np.nanmedian):
        td = self.ds.time.diff(dim=self.obs_dim)
        return pd.Timedelta(average(td))

    def time_to_next(self):
        time = self.ds.time
        lenobs = self.ds.sizes[self.obs_dim]
        td = time.diff(dim=self.obs_dim)
        td = xr.concat((td, td.isel(obs=-1)),
                       dim=self.obs_dim)  # Repeat last time step
        td.coords[self.obs_dim] = time[
            self.obs_dim]  # Same time index as initial
        return td.astype("timedelta64[s]").rename("time_to_next").assign_attrs(
            {"units": "seconds"})

    def is_orthogonal(self):
        return False

    def is_ragged(self):
        return True

    def insert_nan_where(self, condition):
        """
        Insert NaN values in trajectories after given positions, shifting the rest of the trajectory.

        Parameters
        ----------
        condition : xarray.DataArray
            Boolean condition indicating where NaN values should be inserted.

        Returns
        -------
        xarray.Dataset
            Dataset with NaN values inserted at specified positions.
        """
        num_inserts = condition.sum(dim=self.obs_dim)
        max_obs = (self.index_of_last() + 1 + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self.ds.sizes[self.trajectory_dim])
        nd = xr.Dataset(
            coords={
                self.trajectory_dim: ([self.trajectory_dim], trajcoord),
                self.obs_dim:
                ([self.obs_dim], range(max_obs))  # Longest trajectory
            },
            attrs=self.ds.attrs)

        # Add extended variables
        for varname, var in self.ds.data_vars.items():
            if self.obs_dim not in var.dims:
                nd[varname] = var
                continue

            # Create empty DataArray to hold interpolated values for the variable
            da = xr.DataArray(
                data=np.full(
                    (nd.sizes[self.trajectory_dim], nd.sizes[self.obs_dim]),
                    np.nan),
                dims=nd.dims,
                attrs=var.attrs,
            )

            for t in range(self.ds.sizes[
                    self.trajectory_dim]):  # Loop over trajectories
                numins = num_inserts[t]
                olddata = var.isel({self.trajectory_dim: t}).values
                wh = np.argwhere(
                    condition.isel({
                        self.trajectory_dim: t
                    }).values) + 1
                if len(wh) == 0:
                    newdata = olddata
                else:
                    insert_indices = np.concatenate(wh)
                    s = np.split(olddata, insert_indices)

                    if np.issubdtype(var.dtype, np.datetime64):
                        na = np.atleast_1d(np.datetime64("NaT"))
                    else:
                        na = np.atleast_1d(np.nan)
                    newdata = np.concatenate(
                        [np.concatenate((ss, na)) for ss in s])

                newdata = newdata[:max_obs]  # Truncate to max_obs
                da[{
                    self.trajectory_dim: t,
                    self.obs_dim: slice(0, len(newdata))
                }] = newdata

            nd[varname] = da.astype(var.dtype)

        nd = nd.assign_coords({self.obs_dim:
                               np.arange(max_obs)})  # New obs index 1..N

        return nd

    def drop_where(self, condition):
        """
        Remove positions where the condition is True, shifting the rest of the trajectory.

        Parameters
        ----------
        condition : xarray.DataArray
            Boolean condition indicating positions to drop.

        Returns
        -------
        xarray.Dataset
            Dataset with positions removed where the condition is True.
        """
        trajs = []
        newlen = 0
        for i in range(self.ds.sizes[self.trajectory_dim]):
            new = self.ds.isel({
                self.trajectory_dim: i
            }).drop_sel(
                obs=np.where(condition.isel({self.trajectory_dim: i}))[0])
            newlen = max(newlen, new.sizes[self.obs_dim])
            trajs.append(new)

        # Ensure all trajectories have equal length by padding with NaN at the end
        trajs = [
            t.pad(pad_width={
                self.obs_dim: (0, newlen - t.sizes[self.obs_dim])
            }).assign_coords({self.obs_dim:
                              np.arange(newlen)})  # New obs index 1..N
            for t in trajs
        ]

        ds = xr.concat(trajs, dim=self.trajectory_dim, join='exact')

        return ds

    @__require_obs_dim__
    def condense_obs(self) -> xr.Dataset:

        on = self.ds.sizes[self.obs_dim]
        logger.debug(f'Condensing {on} observations.')

        ds = self.ds.copy(deep=True)

        # The observation coordinate will be re-written
        ds = ds.drop_vars([self.obs_dim], errors='ignore')

        assert self.obs_dim in ds[
            self.
            time_varname].dims, "observation not a coordinate of time variable"

        # Move all observations for each trajectory to starting row
        maxN = 0
        for ti in range(len(ds.trajectory)):
            obsvars = [
                var for var in ds.variables if self.obs_dim in ds[var].dims
            ]
            iv = np.full(on, False)
            for var in obsvars:
                ivv = ~pd.isnull(ds[self.time_varname][
                    ti, :])  # valid times in this trajectory.
                iv = np.logical_or(iv, ivv)

            N = np.count_nonzero(iv)
            maxN = max(N, maxN)
            # logger.debug(f'Condensing trajectory {ti=}, observations: {N}..')

            if N == 0:
                logger.error(f'No valid observations in trajectory {ti}.')
                continue

            for var in obsvars:
                # logger.debug(f'Condensing variable {var}..')
                n = np.count_nonzero(~pd.isnull(ds[var][ti, :]))
                assert n <= N, f"Unexpected number of observations in trajectory for {ti=}, {var}: {n} > {N}."

                ds[var][ti, :N] = ds[var][ti, iv]
                ds[var][ti, N:] = np.nan

                # assert (~np.isnan(ds[var][ti, :N])).all(
                # ), "Varying number of valid observations within same trajectory."

        logger.debug(f'Condensed observations from: {on} to {maxN}')
        ds = ds.isel({self.obs_dim: slice(0, maxN)})

        # Write new observation coordinate.
        ds = ds.assign_coords({self.obs_dim: np.arange(0, maxN)})

        return ds

    def append(self, da, obs_dims=None):
        ds = self.ds.copy(deep=True)

        if obs_dims is None:
            obs_dims = [self.obs_dim]
        else:
            obs_dims = list(obs_dims)

        # Increase obs_dims to size of max
        for o in obs_dims:
            print(o)
            N = max(ds.sizes[o], da.sizes[o])
            ds = ds.pad({o: (0, N - ds.sizes[o])})
            da = da.pad({o: (0, N - da.sizes[o])})

        ds = xr.concat((ds, da), dim='trajectory')

        return ds

    def sel(self, *args, **kwargs):
        return self.trajectories().map(lambda d: ensure_time_dim(
            d.traj.to_orthogonal().traj.sel(*args, **kwargs), self.time_varname
        ).traj.to_ragged(self.obs_dim))

    def seltime(self, t0=None, t1=None):
        # Using TrajAn sel method that allows NaN
        return self.trajectories().map(lambda d: ensure_time_dim(
            d.traj.to_orthogonal().traj.seltime(t0, t1), self.time_varname).
                                       traj.to_ragged(self.obs_dim))

    @__require_obs_dim__
    def iseltime(self, i):

        def select(t):
            ii = np.argwhere(~pd.isna(t[self.time_varname]).squeeze())
            ii = ii[i].squeeze() if len(ii) > 0 else ii.squeeze()

            o = t.isel({self.obs_dim: ii})

            if self.obs_dim in o.dims:
                return o
            else:
                return o.expand_dims(self.obs_dim)

        return self.trajectories().map(select)

    def to_orthogonal(self):
        if self.ds.sizes[self.trajectory_dim] > 1:
            raise ValueError(
                "Can not convert a Ragged dataset with multiple trajectories to Orthogonal."
            )
        else:
            ds = self.ds.copy()

            # Do not remove NaN's since these now have meaning
            #ds = ds.dropna(self.obs_dim, how='all')

            # For Orthogonal datasets, we rename obs-dimension to name of time variable
            # so that time becomes a coordinate variable,
            # i.e. typically:   time(traj, obs) -> time(time)
            ds = ds.assign_coords({self.obs_dim: ds[self.time_varname]})
            ds = ds.drop_vars(self.time_varname).rename(
                {self.obs_dim: self.time_varname})
            ds[self.time_varname] = ds[self.time_varname].squeeze(
                self.trajectory_dim)

            # Do not remove NaN's since these now have meaning
            #ds = ds.loc[{self.time_varname: ~pd.isna(ds[self.time_varname])}]
            #_, ui = np.unique(ds[self.time_varname], return_index=True)
            #ds = ds.isel({self.time_varname: ui})

            # Keep trajectory dimension, although always length 1 for single trajectories
            ds = ds.assign_coords(
                {self.trajectory_dim: ds[self.trajectory_dim]})

            return ds

    def to_ragged(self, obs_dim='obs'):
        if self.obs_dim != obs_dim:
            return self.ds.rename({self.obs_dim: obs_dim}).copy()
        else:
            return self.ds.copy()

    @__require_obs_dim__
    def gridtime(self, *args, **kwargs):
        """
        Interpolate the dataset to a given time grid.

        Parameters
        ----------
        times : str, pandas.Timedelta, or numpy.ndarray
            Time grid to interpolate to. If a string or Timedelta, it specifies the interval.
        time_varname : str, optional
            Name of the time variable, by default the dataset's time variable.
        round : bool, optional
            Whether to round the start and end times to the nearest interval, by default True.

        Returns
        -------
        xarray.Dataset
            Dataset interpolated to the specified time grid.
        """
        gridded = self.ds.groupby(
            self.trajectory_dim).map(lambda d: ensure_time_dim(
                d.traj.to_orthogonal().traj.gridtime(*args, **kwargs), self.
                time_varname))

        # TODO: trajectory index should be preserved so that this should not be necessary
        gridded = gridded.assign_coords(
            {self.trajectory_dim: self.ds[self.trajectory_dim]})

        return gridded

    def skill(self):
        raise ValueError('Not implemented for Ragged datasets')

    def filter(self, method='speed', **kwargs) -> xr.Dataset:
        return self.trajectories().map(lambda d: ensure_time_dim(
            d.traj.to_orthogonal().traj.filter(method=method, **kwargs), self.
            time_varname).traj.to_ragged(self.obs_dim))
