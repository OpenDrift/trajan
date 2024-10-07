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

    def __init__(self, ds, obsdim, timedim):
        super().__init__(ds, obsdim, timedim)

    def timestep(self):
        """Time step between observations in seconds."""
        return ((self.ds.time[1] - self.ds.time[0]) /
                np.timedelta64(1, 's')).values

    def is_1d(self):
        return True

    def is_2d(self):
        return False

    def time_to_next(self):
        time_step = self.ds.time[1] - self.ds.time[0]
        return time_step

    def velocity_spectrum(self):

        if self.ds.sizes['trajectory'] > 1:
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
        if self.ds.sizes['trajectory'] > 1:
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
        if self.ds.sizes['trajectory'] != other.sizes['trajectory']:
            raise ValueError(
                f"There must be the same number of trajectories in the two datasets that are compared. This dataset: {self.ds.sizes['trajectory']}, other: {other.sizes['trajectory']}."
            )

        diff = np.max(
            np.abs((self.ds[self.obsdim] -
                    other[other.traj.obsdim]).astype('timedelta64[s]').astype(
                        np.float64)))

        if not np.isclose(diff, 0):
            raise ValueError(
                f"The two datasets must have approximately equal time coordinates, maximum difference: {diff} seconds. Consider using `gridtime` to interpolate one of the datasets."
            )

        s = np.zeros((self.ds.sizes['trajectory']), dtype=np.float32)

        # ds = self.ds.dropna(dim=self.obsdim)
        # other = other.dropna(dim=other.traj.obsdim)

        ds = self.ds.transpose('trajectory', self.obsdim, ...)
        other = other.transpose('trajectory', other.traj.obsdim, ...)

        lon0 = ds.traj.tlon
        lat0 = ds.traj.tlat
        lon1 = other.traj.tlon
        lat1 = other.traj.tlat

        for ti in range(0, len(s)):
            if method == 'liu-weissberg':
                s[ti] = skill.liu_weissberg(lon0.isel(trajectory=ti),
                                            lat0.isel(trajectory=ti),
                                            lon1.isel(trajectory=ti),
                                            lat1.isel(trajectory=ti), **kwargs)
            else:
                raise ValueError(f"Unknown skill-score method: {method}.")

        return xr.DataArray(s,
                            name='Skillscore',
                            coords={'trajectory': self.ds.trajectory},
                            attrs={'method': method})

    def seltime(self, t0=None, t1=None):
        return self.ds.sel({self.timedim: slice(t0, t1)})

    def gridtime(self, times, timedim=None, round=True):
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

        timedim = self.timedim if timedim is None else timedim

        ds = self.ds

        if self.obsdim != timedim:
            ds = ds.rename({
                self.obsdim: timedim
            }).set_index({timedim: timedim})

        _, ui = np.unique(ds[timedim], return_index=True)

        if len(ui) != len(self.ds[timedim]):
            logger.warning('non-unique time points, dropping time-duplicates')

        ds = ds.isel({timedim: ui})
        ds = ds.isel({timedim: np.where(~pd.isna(ds[timedim].values))[0]})

        if ds.sizes[timedim] > 0:
            ds = ds.interp({timedim: times})
        else:
            logger.warning(f"time dimension ({timedim}) is zero size")

        if not 'trajectory' in ds.dims:
            ds = ds.expand_dims('trajectory')

        return ds
