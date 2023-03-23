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

    def skill(self, other, method='liu-weissberg', **kwargs):
        """
        Compare the skill score between this trajectory and `other`.

        Args:

            other: Another trajectory dataset.

            method: skill-score method, currently only liu-weissberg. See :mod:`trajan.skill`.

            **kwargs: passed on to the skill-score method.

        Returns:

            skill: The skill-score in the same dimensions as this dataset.

        .. note::

            The datasets must be sampled (or have observations) at approximately the same timesteps. Consider using :meth:`trajan.traj2d.gridtime` to interpolate one of the datasets to the other.


        .. note::

            The datasets must have the same number of trajectories. If you wish to compare a single trajectory to many others, duplicate it along the trajectory dimension to match the trajectory dimension of the other. See further down for an example.


        .. testcode::

            import xarray as xr
            import trajan as _
            import lzma

            b = lzma.open('examples/barents.nc.xz')
            ds = xr.open_dataset(b)

            other = ds.copy()

            ds = ds.traj.gridtime('1H')

            other = other.traj.gridtime(ds.time)
            skill = ds.traj.skill(other)

            print(skill)

        .. testoutput::

            <xarray.DataArray 'Skill-score' (trajectory: 2)>
            array([1., 1.], dtype=float32)
            Coordinates:
              * trajectory  (trajectory) int64 0 1
            Attributes:
                method:   liu-weissberg


        If you need to broadcast a dataset with a single drifter to one with many you can use `xarray.broadcast` or `xarray.Dataset.broadcast_like`:

        .. note::

            If the other dataset has any other dimensions, on any other variables, you need to exclude those when broadcasting.

        .. testcode::

            b0 = ds.isel(trajectory=0) # `b0` now only has a single drifter (no trajectory dimension)

            b0 = b0.broadcast_like(ds)
            skill = b0.traj.skill(ds)

            print(skill)

        .. testoutput::

            <xarray.DataArray 'Skill-score' (trajectory: 2)>
            array([1.        , 0.60894716], dtype=float32)
            Coordinates:
              * trajectory  (trajectory) int64 0 1
            Attributes:
                method:   liu-weissberg

        """

        if self.ds.dims['trajectory'] != other.dims['trajectory']:
            raise ValueError(
                f"There must be the same number of trajectories in the two datasets that are compared. This dataset: {self.ds.dims['trajectory']}, other: {other.dims['trajectory']}."
            )

        diff = np.max(
            np.abs((self.ds[self.obsdim] -
                    other[other.traj.obsdim]).astype('timedelta64[s]').astype(
                        np.float64)))

        if not np.isclose(diff, 0):
            raise ValueError(
                f"The two datasets must have approximately equal time coordinates, maximum difference: {diff} seconds. Consider using `gridtime` to interpolate one of the datasets."
            )

        s = np.zeros((self.ds.dims['trajectory']), dtype=np.float32)

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
                            name='Skill-score',
                            coords={'trajectory': self.ds.trajectory},
                            attrs={'method': method})
