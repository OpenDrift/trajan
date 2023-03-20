import numpy as np
import xarray as xr
import numpy as np
from .traj import Traj
from . import skill


class Traj1d(Traj):
    """
    A structured dataset, where each trajectory is always given at the same times. Typically the output from a model or from a gridded dataset.
    """

    def __init__(self, ds):
        super().__init__(ds)

    def timestep(self):
        """
        Time step between observations in seconds.
        """
        return ((self.ds.time[1] - self.ds.time[0]) /
                np.timedelta64(1, 's')).values

    def time_to_next(self):
        time_step = self.ds.time[1] - self.ds.time[0]
        return time_step

    def velocity_spectrum(self):

        if self.ds.dims['trajectory'] > 1:
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
        if self.ds.dims['trajectory'] > 1:
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

    def skill(self, other, method='liu-weissberg'):
        """
        Compare the skill score between this trajectory and `other`.

        Args:

            other: Another trajectory dataset.

        Returns:

            skill: The skill-score in the same dimensions as this dataset.

        Observations only consisting of `NaN`s will be dropped using `xarray.Dataset.dropna` before comparison.

        The datasets must be sampled (or have observations) at approximately the same timesteps. Consider using :meth:`trajan.traj2d.gridtime` to interpolate one of the datasets to the other.

        The datasets must have the same number of trajectories. If you wish to compare a single trajectory to many others, duplicate it along the trajectory dimension to match the trajectory dimension of the other.


        >>> ds1 = ds1.traj.gridtime('1H')
        >>> other = other.traj.gridtime(ds1.time)
        >>> skill = ds1.traj.skill(other)
        >>> print(skill)
        """

        if self.ds.dims['trajectory'] != other.dims['trajectory']:
            raise ValueError(
                f"There must be the same number of trajectories in the two datasets that are compared. This dataset: {self.ds.dims['trajectory']}, other: {other.dims['trajectory']}."
            )

        diff = np.max(
            np.abs(
                (self.ds['time'] -
                 other['time']).astype('timedelta64[s]').astype(np.float64)))

        if not np.isclose(diff, 0):
            raise ValueError(
                f"The two datasets must have approximately equal time coordinates, maximum difference: {diff} seconds. Consider using `gridtime` to interpolate one of the datasets."
            )

        s = np.zeros((self.ds.dims['trajectory']), dtype=np.float32)

        ds = self.ds.dropna(dim=self.obsdim)
        other = other.dropna(dim=other.traj.obsdim)

        lon0 = ds.traj.tlon
        lat0 = ds.traj.tlat
        lon1 = other.traj.tlon
        lat1 = other.traj.tlat

        for ti in range(0, len(s)):
            if method == 'liu-weissberg':
                s[ti] = skill.liu_weissberg(lon0, lat0, lon1, lat1)
            else:
                raise ValueError(f"Unknown skill-score method: {method}.")

        return xr.DataArray(s,
                            name='Skill-score',
                            coords={'trajectory': self.ds.trajectory},
                            attrs={'method': method})
