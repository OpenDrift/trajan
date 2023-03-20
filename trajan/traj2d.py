import xarray as xr
import numpy as np

from .traj import Traj

class Traj2d(Traj):
    """
    A unstructured dataset, where each trajectory may have observations at different times. Typically from a collection of drifters.
    """

    def __init__(self, ds):
        super().__init__(ds)


    def timestep(self, average=np.nanmedian):
        """
        Return median time step between observations in seconds.
        """
        td = np.diff(self.ds.time, axis=1) / np.timedelta64(1, 's')
        td = average(td)
        return td

    def time_to_next(self):
        """Returns time from one position to the next

           Returned datatype is np.timedelta64
           Last time is repeated for last position (which has no next position)
        """
        time = self.ds.time
        lenobs = self.ds.dims['obs']
        td = time.isel(obs=slice(1, lenobs)) - time.isel(
            obs=slice(0, lenobs - 1))
        td = xr.concat((td, td.isel(obs=-1)),
                       dim='obs')  # repeating last time step
        return td

    def insert_nan_where(self, condition):
        """Insert NaN-values in trajectories after given positions, shifting rest of trajectory"""

        index_of_last = self.index_of_last()
        num_inserts = condition.sum(dim='obs')
        max_obs = (index_of_last + num_inserts).max().values

        # Create new empty dataset with extended obs dimension
        trajcoord = range(self.ds.dims['trajectory'])
        nd = xr.Dataset(
            coords={
                'trajectory': (["trajectory"], range(self.ds.dims['trajectory'])),
                'obs': (['obs'], range(max_obs))  # Longest trajectory
            },
            attrs=self.ds.attrs)

        # Add extended variables
        for varname, var in self.ds.data_vars.items():
            if 'obs' not in var.dims:
                nd[varname] = var
                continue
            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(
                data=np.zeros(tuple(nd.dims[di] for di in nd.dims)) * np.nan,
                dims=nd.dims,
                attrs=var.attrs,
            )

            for t in range(
                    self.ds.dims['trajectory']):  # loop over trajectories
                numins = num_inserts[t]
                olddata = var.isel(trajectory=t).values
                wh = np.argwhere(condition.isel(trajectory=t).values) + 1
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

                newdata = newdata[slice(0, max_obs -
                                        1)]  # truncating, should be checked
                da[{'trajectory': t, 'obs': slice(0, len(newdata))}] = newdata

            nd[varname] = da.astype(var.dtype)

        nd = nd.drop_vars(('obs', 'trajectory'))  # Remove coordinates

        return nd

    def drop_where(self, condition):
        """Remove positions where condition is True, shifting rest of trajectory"""

        trajs = []
        newlen = 0
        for i in range(self.ds.dims['trajectory']):
            new = self.ds.isel(trajectory=i).drop_sel(obs=np.where(
                condition.isel(
                    trajectory=i))[0])  # Dropping from given trajectory
            newlen = max(newlen, new.dims['obs'])
            trajs.append(new)

        # Ensure all trajectories have equal length, by padding with NaN at end
        trajs = [
            t.pad(pad_width={'obs': (0, newlen - t.dims['obs'])})
            for t in trajs
        ]

        return xr.concat(trajs, dim='trajectory')

