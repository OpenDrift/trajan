import pyproj
import numpy as np
import xarray as xr

class Traj:
    ds: xr.Dataset

    def __init__(self, ds):
        self.ds = ds

        if 'obs' in self.ds.dims:
            self.obsdim = 'obs'
        elif 'time' in self.ds.dims:
            self.obsdim = 'time'
        else:
            raise ValueError('No time or obs dimension')

    def index_of_last(self):
        """Find index of last valid position along each trajectory"""
        return np.ma.notmasked_edges(np.ma.masked_invalid(self.ds.lon.values),
                                     axis=1)[1][1]

    def speed(self):
        """Returns the speed [m/s] along trajectories"""

        distance = self.distance_to_next()
        timedelta_seconds = self.time_to_next() / np.timedelta64(1, 's')

        return distance / timedelta_seconds

    def distance_to_next(self):
        """Returns distance in m from one position to the next

           Last time is repeated for last position (which has no next position)
        """

        lon = self.ds.lon
        lat = self.ds.lat
        lenobs = self.ds.dims[self.obsdim]
        lonfrom = lon.isel({self.obsdim: slice(0, lenobs - 1)})
        latfrom = lat.isel({self.obsdim: slice(0, lenobs - 1)})
        lonto = lon.isel({self.obsdim: slice(1, lenobs)})
        latto = lat.isel({self.obsdim: slice(1, lenobs)})
        geod = pyproj.Geod(ellps='WGS84')
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto, latto)

        distance = xr.DataArray(distance, coords=lonfrom.coords, dims=lon.dims)
        distance = xr.concat((distance, distance.isel({self.obsdim: -1})),
                             dim=self.obsdim)  # repeating last time step to
        return distance

    def azimuth_to_next(self):
        """Returns azimution travel direction in degrees from one position to the next

           Last time is repeated for last position (which has no next position)
        """

        # TODO: method is almost duplicate of "distance_to_next" above
        lon = self.ds.lon
        lat = self.ds.lat
        lenobs = self.ds.dims[self.obsdim]
        lonfrom = lon.isel({self.obsdim: slice(0, lenobs - 1)})
        latfrom = lat.isel({self.obsdim: slice(0, lenobs - 1)})
        lonto = lon.isel({self.obsdim: slice(1, lenobs)})
        latto = lat.isel({self.obsdim: slice(1, lenobs)})
        geod = pyproj.Geod(ellps='WGS84')
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto, latto)

        azimuth_forward = xr.DataArray(azimuth_forward, coords=lonfrom.coords, dims=lon.dims)
        azimuth_forward = xr.concat((azimuth_forward, azimuth_forward.isel({self.obsdim: -1})),
                             dim=self.obsdim)  # repeating last time step to
        return azimuth_forward

    def velocity_components(self):
        """Returns velocity components [m/s] from one position to the next

           Last time is repeated for last position (which has no next position)
        """
        speed = self.speed()
        azimuth = self.azimuth_to_next()

        u = speed*np.cos(azimuth)
        v = speed*np.sin(azimuth)

        return u, v
