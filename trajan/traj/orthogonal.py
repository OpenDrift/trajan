import numpy as np
import xarray as xr
import pandas as pd
import logging
import pyproj
from . import Traj, inherit_docstrings
from .. import skill

logger = logging.getLogger(__name__)


def _nsigma_sliding_filter(arr, nsigma=5.0, side_half_width=2):
    """Apply a sliding n-sigma outlier filter to a 1D numpy array.

    For each central point at index ``i``, if that point deviates more than
    ``nsigma`` standard deviations from the mean of its neighbours in
    ``[i-side_half_width, i+side_half_width]`` (excluding itself), it is set to NaN.
    Points within ``side_half_width`` of either end are left unchanged.

    Parameters
    ----------
    arr : numpy.ndarray, shape (N,)
        Input 1-D array.
    nsigma : float
        Number of standard deviations for the outlier threshold. Default: 5.
    side_half_width : int
        Number of neighbours on each side of the central point. Default: 2.

    Returns
    -------
    numpy.ndarray
        Copy of the input array with outliers replaced by NaN.
    """
    arr = np.array(arr, dtype=float)
    n = len(arr)
    for i in range(side_half_width, n - side_half_width):
        neighbours = np.concatenate([arr[i - side_half_width:i],
                                     arr[i + 1:i + side_half_width + 1]])
        mean = np.mean(neighbours)
        std = np.std(neighbours)
        if np.abs(arr[i] - mean) > nsigma * std:
            arr[i] = np.nan
    return arr


@inherit_docstrings
class TrajOrthogonal(Traj):
    """
    A structured dataset, where each trajectory is always given at the same times. Typically the output from a model or from a gridded dataset.
    """

    def __init__(self, ds, trajectory_dim, obs_dim, time_varname):
        super().__init__(ds, trajectory_dim, obs_dim, time_varname)

    def timestep(self):  # TODO: assumes constant timestep, but could be variable
        return pd.Timedelta((self.ds.time[1] - self.ds.time[0]).values)

    def is_orthogonal(self):
        return True

    def is_ragged(self):
        return False

    def to_ragged(self, obs_dim='obs'):
        ds = self.ds.copy()
        time = ds[self.time_varname].rename({self.time_varname: obs_dim}).expand_dims(
            dim={self.trajectory_dim: ds.sizes[self.trajectory_dim]}
        ).assign_coords({self.trajectory_dim: ds[self.trajectory_dim]})
        ds = ds.rename({self.time_varname: obs_dim})
        ds[self.time_varname] = time
        ds[obs_dim] = xr.DataArray(np.arange(0, ds.sizes[obs_dim]), dims=[obs_dim])
        return ds

    def to_orthogonal(self):
        return self.ds.copy()

    def time_to_next(self):
        time_step = self.ds.time[1] - self.ds.time[0]
        return xr.DataArray(time_step, name="time_to_next", attrs={"units": "seconds"})

    def velocity_spectrum(self):
        if self.ds.sizes[self.trajectory_dim] > 1:
            raise ValueError('Spectrum can only be calculated for a single trajectory')

        u, v = self.velocity_components()
        u = u.squeeze()
        v = v.squeeze()
        u = u[np.isfinite(u)]
        v = v[np.isfinite(v)]

        timestep_h = (self.ds.time[1] - self.ds.time[0]) / np.timedelta64(1, 'h')  # hours since start

        ps = np.abs(np.fft.rfft(np.abs(u + 1j * v)))
        freq = np.fft.rfftfreq(n=u.size, d=timestep_h.values)
        freq[0] = np.nan

        da = xr.DataArray(
            data=ps,
            name='velocity_spectrum',
            dims=['period'],
            coords={'period': (['period'], 1 / freq, {'units': 'hours'})},
            attrs={'units': 'power'}
        )

        return da

    def rotary_spectrum(self):
        """Calculate the rotary spectrum for a single trajectory.

        .. note:: This method is not yet fully implemented.
        """
        ### TODO unfinished method

        from .tools import rotary_spectra
        if self.ds.sizes[self.trajectory_dim] > 1:
            raise ValueError(
                'Spectrum can only be calculated for a single trajectory')

        u, v = self.velocity_components()
        u = u.squeeze()
        v = v.squeeze()
        u = u[np.isfinite(u)]
        v = v[np.isfinite(v)]

        puv, quv, cw, ccw, F = rotary_spectra(u, v)
        print(puv)
        import matplotlib.pyplot as plt
        plt.plot(1 / F, cw)
        plt.xlim([0, 30])
        plt.show()

    def skill_along_trajectory(self, expected, **kwargs):
        """_Skill score is calculated for each trajectory versus the matcing part of the expected (single) trajectory.
        """

        expected = expected.traj  # Normalise

        numtraj_expected = expected.ds.sizes[expected.trajectory_dim]
        if numtraj_expected > 1:
            raise ValueError(
                f'Expected must contain a single trajectory, this contains {numtraj_expected}.'
            )

        start_lon = self.ds.lon.bfill(dim='time').isel(time=0)
        start_lat = self.ds.lat.bfill(dim='time').isel(time=0)
        end_lon = self.ds.lon.ffill(dim='time').isel(time=-1)
        end_lat = self.ds.lat.ffill(dim='time').isel(time=-1)

        def skill_matching(traj, expected):
            traj = traj.where(np.isfinite(traj.lon), drop=True)
            expected_overlap = expected.sel(
                time=slice(traj.time[0], traj.time[-1]))
            mask = False
            if traj.sizes[self.obs_dim] != expected_overlap.sizes[
                    expected_overlap.traj.obs_dim]:
                traj = traj.sel(time=slice(expected_overlap.time[0],
                                           expected_overlap.time[-1]))
                mask = True
            s = traj.traj.skill(expected_overlap, **kwargs)
            if mask is True:
                s = s * np.nan  # Mask out the skill score if the trajectories are not of equal length
            s['start_time'] = traj.time[0]
            s['end_time'] = traj.time[-1]
            return s

        s = self.ds.groupby(self.trajectory_dim).apply(skill_matching,
                                                       expected=expected)
        s['start_lon'] = start_lon
        s['start_lat'] = start_lat
        s['end_lon'] = end_lon
        s['end_lat'] = end_lat
        s = s.drop_vars('time')

        return s

    def skill(self, expected, method='liu-weissberg', **kwargs) -> xr.DataArray:
        expected = expected.traj  # Normalize
        expected_trajdim = expected.trajectory_dim
        self_trajdim = self.trajectory_dim

        numtraj_self = self.ds.sizes[self.trajectory_dim]
        numtraj_expected = expected.ds.sizes[expected.trajectory_dim]
        if numtraj_self > 1 and numtraj_expected > 1 and numtraj_self != numtraj_expected:
            raise ValueError(
                'Datasets must have the same number of trajectories, or a single trajectory. '
                f'This dataset: {numtraj_self}, expected: {numtraj_expected}.'
            )

        numobs_self = self.ds.sizes[self.obs_dim]
        numobs_expected = expected.ds.sizes[expected.obs_dim]
        if numobs_self != numobs_expected:
            raise ValueError(
                f'Trajectories must have the same lengths. This dataset: {numobs_self}, expected: {numobs_expected}.'
            )

        diff = np.max(
            np.abs((
                self.ds[self.obs_dim] -
                expected.ds[expected.obs_dim]).astype('timedelta64[s]').astype(
                    np.float64)))
        if not np.isclose(diff, 0):
            raise ValueError(
                f"The two datasets must have approximately equal time coordinates, maximum difference: {diff} seconds. Consider using `gridtime` to interpolate one of the datasets."
            )

        # Skillscore methods expect that obs_dim is the first dimension
        ds = self.ds.transpose(self.obs_dim, ...)
        expected = expected.ds.transpose(expected.obs_dim, ...)

        # Broadcast so that we have same dimensions in both datasets
        if numtraj_expected == 1 and expected_trajdim in expected.sizes:
            expected = expected.squeeze(dim=expected_trajdim, drop=True)
        elif numtraj_self == 1 and self_trajdim in ds.sizes:
            ds = ds.squeeze(dim=self_trajdim, drop=True)
        ds = ds.broadcast_like(expected)
        expected = expected.broadcast_like(ds)

        # Skillscore methods expect that obs_dim is the first dimension
        ds = ds.transpose(self.obs_dim, ...)
        expected = expected.transpose(expected.traj.obs_dim, ...)

        if method == 'liu-weissberg':
            skill_method = skill.liu_weissberg
        elif method == 'darpa':
            skill_method = skill.darpa
        else:
            raise ValueError(f"Unknown skill-score method: {method}.")

        s = skill_method(expected.traj.tlon, expected.traj.tlat, ds.traj.tlon,
                         ds.traj.tlat, **kwargs)

        if kwargs.get('cumulative', False):
            # Build DataArray with explicit dims from the (broadcasted) ds,
            # then drop any size-1 dims that are not in self, and reorder to
            # match self.ds.lon dimension order.
            result = xr.DataArray(s, dims=ds.lon.dims, coords=ds.lon.coords,
                                  name='Skillscore', attrs={'method': method})
            self_dims = set(self.ds.lon.dims)
            for dim in list(result.dims):
                if dim not in self_dims and result.sizes[dim] == 1:
                    result = result.squeeze(dim=dim, drop=True)
            ordered = [d for d in self.ds.lon.dims if d in result.dims]
            extra = [d for d in result.dims if d not in ordered]
            return result.transpose(*(ordered + extra))

        newcoords = dict(ds.lon.sizes)
        newcoords.pop('time')
        for dim in newcoords:
            if dim in ds.coords:
                newcoords[dim] = dict(ds.coords)[dim]
            else:
                newcoords[dim] = np.arange(dict(newcoords)[dim])
        return xr.DataArray(s,
                            name='Skillscore',
                            coords=newcoords,
                            attrs={'method': method})

    def sel(self, *args, **kwargs):
        return self.ds.sel(*args, **kwargs)

    def seltime(self, t0=None, t1=None):
        # Preserving NaN in trajectories, as these provide information about gaps / segments
        subset_indices = np.where((self.ds[self.time_varname] >= pd.to_datetime(t0)) &
                                  (self.ds[self.time_varname] <= pd.to_datetime(t1)))[0]
        return self.ds.isel({self.obs_dim: slice(subset_indices.min(), subset_indices.max()+1)})

    def iseltime(self, i):
        return self.ds.isel({self.time_varname: i})

    def distance_to(self, other) -> xr.Dataset:
        other = other.broadcast_like(self.ds)

        geod = pyproj.Geod(ellps='WGS84')
        az_fwd, a2, distance = geod.inv(self.tlon, self.tlat, other.traj.tlon,
                                        other.traj.tlat)

        ds = xr.Dataset()
        ds['distance'] = xr.DataArray(distance,
                                      name='distance',
                                      coords=self.tlon.coords,
                                      attrs={'units': 'm'})

        ds['az_fwd'] = xr.DataArray(az_fwd,
                                    name='forward azimuth',
                                    coords=self.tlon.coords,
                                    attrs={'units': 'degrees'})

        ds['az_bwd'] = xr.DataArray(a2,
                                    name='back azimuth',
                                    coords=self.tlon.coords,
                                    attrs={'units': 'degrees'})

        return ds

    def gridtime(self, times, time_varname=None, round=True):
        if isinstance(times, str) or isinstance(
                times, pd.Timedelta):  # Make time series with given interval
            if round is True:
                start_time = np.nanmin(np.asarray(
                    self.ds.time.dt.floor(times)))
                end_time = np.nanmax(np.asarray(self.ds.time.dt.ceil(times)))
            else:
                start_time = np.nanmin(np.asarray(self.ds.time))
                end_time = np.nanmax(np.asarray(self.ds.time))
            times = pd.date_range(start_time,
                                  end_time,
                                  freq=times,
                                  inclusive='both')

        if not isinstance(times, np.ndarray):
            times = times.to_numpy()

        time_varname = self.time_varname if time_varname is None else time_varname
        times = times.astype(self.ds[time_varname].dtype)  # Make sure we have same dtype for ds and times array

        ds = self.ds.copy()

        # We fill NaN-values in time coordinate to be able to interpolate
        ds[time_varname] = ds[time_varname].interpolate_na(
                dim=time_varname, method='linear', fill_value='extrapolate', use_coordinate=False)

        #if self.obs_dim != time_varname:  # TODO: can this ever happen?
        #    ds = ds.rename({self.obs_dim: time_varname}).set_index({time_varname: time_varname})

        _, ui = np.unique(ds[time_varname], return_index=True)
        if len(ui) != len(self.ds[time_varname]):
            logger.warning('non-unique time points, dropping time-duplicates')
        ds = ds.isel({time_varname: ui})
        #ds = ds.isel(
        #    {time_varname: np.where(~pd.isna(ds[time_varname].values))[0]})

        if ds.sizes[time_varname] > 0:
            ds = ds.interp({time_varname: times})
        else:
            logger.warning(f"time variable ({time_varname}) has zero size")

        if not 'trajectory' in ds.dims:
            ds = ds.expand_dims('trajectory')

        return ds

    def trim(self):
        """
        Remove empty positions at start and end of trajectory.

        Returns
        -------
        xarray.Dataset
            Trimmed dataset
        """

        mask = np.isfinite(self.tlon)
        first = mask.argmax(dim='time')
        last  = mask.sizes['time'] - 1 - mask.isel(time=slice(None, None, -1)).argmax(dim="time")
        firstindex = first.min().values
        lastindex = last.max().values
        numobs = self.ds.sizes[self.obs_dim]
        logger.debug(f'Trimming {firstindex} points from start and {numobs - lastindex} points from end')
        return self.ds.isel(time=slice(firstindex, lastindex))

    def filter(self, method='speed', max_speed=10., nsigma=5.0, side_half_width=2) -> xr.Dataset:
        lon_name = self.tx.name
        lat_name = self.ty.name

        if method == 'speed':
            # Walk through non-NaN positions in order, comparing each to the
            # last accepted "good" position.  When a position is too far away
            # (speed > max_speed) it is masked and the "last good" pointer is
            # NOT advanced, so the next position is still compared to the same
            # good baseline.  This correctly clears entire runs of stuck/invalid
            # GPS readings (e.g. no-fix sentinel values at (0,0)) in a single
            # O(N) pass without falsely masking the valid positions that bracket
            # the bad run.
            geod = pyproj.Geod(ellps='WGS84')
            ds = self.ds.copy(deep=True)
            n_traj = ds.sizes[self.trajectory_dim] if self.trajectory_dim else 1

            for ti in range(n_traj):
                if self.trajectory_dim:
                    lons  = ds[lon_name].values[ti].copy().astype(float)
                    lats  = ds[lat_name].values[ti].copy().astype(float)
                else:
                    lons  = ds[lon_name].values.copy().astype(float)
                    lats  = ds[lat_name].values.copy().astype(float)
                times = ds[self.time_varname].values  # 1D, shared across trajectories

                valid = ~(np.isnan(lons) | np.isnan(lats) | pd.isnull(times))
                indices = np.where(valid)[0]
                if len(indices) < 2:
                    continue

                prev = indices[0]
                for idx in indices[1:]:
                    dt = (times[idx] - times[prev]) / np.timedelta64(1, 's')
                    if dt <= 0:
                        lons[idx] = np.nan
                        lats[idx] = np.nan
                        continue
                    _, _, dist = geod.inv(lons[prev], lats[prev], lons[idx], lats[idx])
                    if dist / dt > max_speed:
                        lons[idx] = np.nan
                        lats[idx] = np.nan
                        # keep prev unchanged — next position compares to same baseline
                    else:
                        prev = idx

                if self.trajectory_dim:
                    ds[lon_name].values[ti] = lons
                    ds[lat_name].values[ti] = lats
                else:
                    ds[lon_name].values[:] = lons
                    ds[lat_name].values[:] = lats

            return ds

        elif method == 'nsigma_sliding':
            ds = self.ds.copy(deep=True)
            n_traj = ds.sizes[self.trajectory_dim] if self.trajectory_dim else 1
            for ti in range(n_traj):
                if self.trajectory_dim:
                    ds[lat_name].values[ti] = _nsigma_sliding_filter(
                        ds[lat_name].values[ti], nsigma, side_half_width)
                    ds[lon_name].values[ti] = _nsigma_sliding_filter(
                        ds[lon_name].values[ti], nsigma, side_half_width)
                else:
                    ds[lat_name].values[:] = _nsigma_sliding_filter(
                        ds[lat_name].values, nsigma, side_half_width)
                    ds[lon_name].values[:] = _nsigma_sliding_filter(
                        ds[lon_name].values, nsigma, side_half_width)
            return ds

        else:
            raise ValueError(
                f"Unknown filter method: '{method}'. Choose 'speed' or 'nsigma_sliding'."
            )
