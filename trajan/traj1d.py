import numpy as np
import scipy as sp
import xarray as xr
from .traj import Traj


class Traj1d(Traj):
    """
    A structured dataset, where each trajectory is always given at the same times. Typically the output from a model or from a gridded dataset.
    """

    def __init__(self, ds):
        super().__init__(ds)


    def time_to_next(self):
        time_step = self.ds.time[1] - self.ds.time[0]
        return time_step

    def velocity_spectrum(self):

        if self.ds.dims['trajectory'] > 1:
            raise ValueError('Spectrum can only be calculated for a single trajectory')

        u, v = self.velocity_components()
        u = u.squeeze()
        v = v.squeeze()
        u = u[np.isfinite(u)]
        v = v[np.isfinite(v)]

        timestep_h = (self.ds.time[1] - self.ds.time[0]) / np.timedelta64(1, 'h')  # hours since start

        ps = np.abs(np.fft.rfft(np.abs(u + 1j*v)))
        freq = np.fft.rfftfreq(n=u.size, d=timestep_h.values)
        freq[0] = np.nan

        da = xr.DataArray(
            data=ps, name='velocity spectrum',
            dims=['period'],
            coords={'period': (['period'], 1/freq, {'units': 'hours'})},
            attrs={'units': 'power'}
            )

        return da
