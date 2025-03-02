import numpy as np
import xarray as xr
import numpy as np
import pandas as pd
import logging
import pyproj
from .traj import Traj
from . import skill

logger = logging.getLogger(__name__)


class Traj1d(Traj):
    """
    A structured dataset, where each trajectory is always given at the same times. Typically the output from a model or from a gridded dataset.
    """

    def __init__(self, ds, trajectory_dim, obs_dim, time_varname):
        super().__init__(ds, trajectory_dim, obs_dim, time_varname)

    def timestep(self):
        """Time step between observations in seconds."""
        return ((self.ds.time[1] - self.ds.time[0]) /
                np.timedelta64(1, 's')).values

    def is_1d(self):
        return True

    def is_2d(self):
        return False

    def to_2d(self, obs_dim='obs'):
        ds = self.ds.copy()
        time = ds[self.time_varname].rename({
            self.time_varname: obs_dim
        }).expand_dims(
            dim={self.trajectory_dim: ds.sizes[self.trajectory_dim]})
        # TODO should also add cf_role here
        ds = ds.rename({self.time_varname: obs_dim})
        ds[self.time_varname] = time
        ds[obs_dim] = np.arange(0, ds.sizes[obs_dim])

        return ds

    def to_1d(self):
        return self.ds.copy()

    def time_to_next(self):
        time_step = self.ds.time[1] - self.ds.time[0]
        return time_step

    def velocity_spectrum(self):

        if self.ds.sizes[self.trajectory_dim] > 1:
            raise ValueError(
                'Spectrum can only be calculated for a single trajectory')

        u, v = self.velocity_components()
        u = u.squeeze()
        v = v.squeeze()
        u = u[np.isfinite(u)]
        v = v[np.isfinite(v)]

        timestep_h = (self.ds.time[1] - self.ds.time[0]) / np.timedelta64(
            1, 'h')  # hours since start

        ps = np.abs(np.fft.rfft(np.abs(u + 1j * v)))
        freq = np.fft.rfftfreq(n=u.size, d=timestep_h.values)
        freq[0] = np.nan

        da = xr.DataArray(
            data=ps,
            name='velocity spectrum',
            dims=['period'],
            coords={'period': (['period'], 1 / freq, {
                'units': 'hours'
            })},
            attrs={'units': 'power'})

        return da

    def rotary_spectrum(self):
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

    def skill(self, other, method='liu-weissberg', **kwargs):

        # Broadcast so that we have same dimensions in both datasets
        other = other.broadcast_like(self.ds)
        ds = self.ds.broadcast_like(other)

        other = other.traj  # Normalise

        numtraj_self = self.ds.sizes[self.trajectory_dim]
        numtraj_other = other.ds.sizes[other.trajectory_dim]
        if numtraj_self > 1 and numtraj_other > 1 and numtraj_self != numtraj_other:
            raise ValueError(
                'Datasets must have the same number of trajectories, or a single trajectory. '
                f'This dataset: {numtraj_self}, other: {numtraj_other}.'
            )

        numobs_self = self.ds.sizes[self.obs_dim]
        numobs_other = other.ds.sizes[other.obs_dim]
        if numobs_self != numobs_other:
            raise ValueError(
                f'Trajectories must have the same lengths. This dataset: {numobs_self}, other: {numobs_other}.'
            )

        diff = np.max(
            np.abs((self.ds[self.obs_dim] -
                    other.ds[other.obs_dim]).astype('timedelta64[s]').astype(
                        np.float64)))
        if not np.isclose(diff, 0):
            raise ValueError(
                f"The two datasets must have approximately equal time coordinates, maximum difference: {diff} seconds. Consider using `gridtime` to interpolate one of the datasets."
            )

        # ds = self.ds.dropna(dim=self.obs_dim)
        # other = other.dropna(dim=other.traj.obs_dim)

        # Skillscore methods expect that obs_dim is the first dimension
        ds = ds.transpose(self.obs_dim, ...)
        other = other.ds.transpose(other.obs_dim, ...)

        if method == 'liu-weissberg':
            skill_method = skill.liu_weissberg
        else:
            raise ValueError(f"Unknown skill-score method: {method}.")
        
        s = skill_method(ds.traj.tlon, ds.traj.tlat, other.traj.tlon, other.traj.tlat, **kwargs)

        newcoords = dict(ds.traj.tlon.sizes)
        newcoords.pop('time')
        for dim in newcoords:
            if dim in ds.coords:
                newcoords[dim] = ds.coords[dim]
            else:
                newcoords[dim] = np.arange(newcoords[dim])

        return xr.DataArray(s,
                            name='Skillscore',
                            coords=newcoords,
                            attrs={'method': method})

    def sel(self, *args, **kwargs):
        return self.ds.sel(*args, **kwargs)

    def seltime(self, t0=None, t1=None):
        return self.ds.sel({self.time_varname: slice(t0, t1)})

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

        ds = self.ds

        if self.obs_dim != time_varname:
            ds = ds.rename({
                self.obs_dim: time_varname
            }).set_index({time_varname: time_varname})

        _, ui = np.unique(ds[time_varname], return_index=True)

        if len(ui) != len(self.ds[time_varname]):
            logger.warning('non-unique time points, dropping time-duplicates')

        ds = ds.isel({time_varname: ui})
        ds = ds.isel(
            {time_varname: np.where(~pd.isna(ds[time_varname].values))[0]})

        if ds.sizes[time_varname] > 0:
            ds = ds.interp({time_varname: times})
        else:
            logger.warning(f"time variable ({time_varname}) has zero size")

        if not 'trajectory' in ds.dims:
            ds = ds.expand_dims('trajectory')

        return ds
