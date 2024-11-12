import numpy as np
import xarray as xr
import numpy as np
import pandas as pd
import logging
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
        }).expand_dims(dim={self.trajectory_dim: ds.sizes[self.trajectory_dim]})
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
        if self.ds.sizes[self.trajectory_dim] != other.sizes[other.traj.trajectory_dim]:
            raise ValueError(
                f"There must be the same number of trajectories in the two datasets that are compared. This dataset: {self.ds.sizes[self.trajectory_dim]}, other: {other.sizes[other.traj.trajectory_dim]}."
            )

        diff = np.max(
            np.abs((self.ds[self.obs_dim] -
                    other[other.traj.obs_dim]).astype('timedelta64[s]').astype(
                        np.float64)))

        if not np.isclose(diff, 0):
            raise ValueError(
                f"The two datasets must have approximately equal time coordinates, maximum difference: {diff} seconds. Consider using `gridtime` to interpolate one of the datasets."
            )

        s = np.zeros((self.ds.sizes[self.trajectory_dim]), dtype=np.float32)

        # ds = self.ds.dropna(dim=self.obs_dim)
        # other = other.dropna(dim=other.traj.obs_dim)

        ds = self.ds.transpose(self.trajectory_dim, self.obs_dim, ...)
        other = other.transpose(other.traj.trajectory_dim, other.traj.obs_dim, ...)

        lon0 = ds.traj.tlon  # TODO should be self.tlon ?
        lat0 = ds.traj.tlat
        lon1 = other.traj.tlon
        lat1 = other.traj.tlat

        for ti in range(0, len(s)):
            if method == 'liu-weissberg':
                s[ti] = skill.liu_weissberg(lon0.isel({self.trajectory_dim: ti}),
                                            lat0.isel({self.trajectory_dim: ti}),
                                            lon1.isel({self.trajectory_dim: ti}),
                                            lat1.isel({self.trajectory_dim: ti}), **kwargs)
            else:
                raise ValueError(f"Unknown skill-score method: {method}.")

        return xr.DataArray(s,
                            name='Skillscore',
                            coords={self.trajectory_dim: self.ds.trajectory},
                            attrs={'method': method})

    def seltime(self, t0=None, t1=None):
        return self.ds.sel({self.time_varname: slice(t0, t1)})

    def iseltime(self, i):
        return self.ds.isel({self.time_varname: i})

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
        ds = ds.isel({time_varname: np.where(~pd.isna(ds[time_varname].values))[0]})

        if ds.sizes[time_varname] > 0:
            ds = ds.interp({time_varname: times})
        else:
            logger.warning(f"time variable ({time_varname}) has zero size")

        if not 'trajectory' in ds.dims:
            ds = ds.expand_dims('trajectory')

        return ds
