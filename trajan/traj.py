import pyproj
import numpy as np
import xarray as xr
import cf_xarray as _
import logging

logger = logging.getLogger(__name__)


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

    def tx(self):
        """
        Trajectory x coordinates (e.g. longitude).
        """
        return self.ds.lon

    def ty(self):
        """
        Trajectory y coordinates (e.g. latitude).
        """
        return self.ds.lat

    @property
    def cartopy_crs(self):
        """
        Retrieve the Cartopy CRS projection from the CF-defined grid-mapping in the dataset.
        """
        raise NotImplemented()

    @property
    def crs(self):
        """
        Retrieve the Proj.4 CRS from the CF-defined grid-mapping in the dataset.
        """
        if len(self.ds.cf.grid_mapping_names) == 0:
            logger.debug(
                f'No grid-mapping specified, checking if coordinates are lon/lat..'
            )
            if self.tx().name == 'lon' or self.tx().name == 'longitude':
                # assume this is in latlon projection
                return pyproj.CRS.from_epsg(4326)
            else:
                # un-projected, assume coordinates are in xy-Cartesian coordinates.
                logger.debug(
                    'Assuming trajectories are in Cartesian coordinates.')
                return None

        else:
            gm = self.ds.cf['grid_mapping']
            logger.debug(f'Constructing CRS from grid_mapping: {gm}')
            return pyproj.CRS.from_cf(gm.attrs)

    def set_crs(self, crs):
        """
        Set and modify the CF-supported grid-mapping / projection in the dataset.
        """

        # TODO: Ideally this would be handled by cf-xarray or rio-xarray.

        gm = crs.to_cf()

        # Create grid mapping variable
        v = xr.DataArray(name=gm['grid_mapping_name'])
        v.attrs = gm
        self.ds[v.name] = v
        self.tx().attrs['grid_mapping'] = v.name
        self.ty().attrs['grid_mapping'] = v.name

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
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto,
                                                 latto)

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
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto,
                                                 latto)

        azimuth_forward = xr.DataArray(azimuth_forward,
                                       coords=lonfrom.coords,
                                       dims=lon.dims)
        azimuth_forward = xr.concat(
            (azimuth_forward, azimuth_forward.isel({self.obsdim: -1})),
            dim=self.obsdim)  # repeating last time step to
        return azimuth_forward

    def velocity_components(self):
        """Returns velocity components [m/s] from one position to the next

           Last time is repeated for last position (which has no next position)
        """
        speed = self.speed()
        azimuth = self.azimuth_to_next()

        u = speed * np.cos(azimuth)
        v = speed * np.sin(azimuth)

        return u, v

    def get_area_convex_hull(self):
        """Returns the area of the convex hull spanned by all particles, per timestep"""

        from scipy.spatial import ConvexHull
        from pyproj import Geod

        area = []
        lons = self.ds.lon.where(self.ds.status == 0)
        lats = self.ds.lat.where(self.ds.status == 0)
        for i in range(self.ds.dims['time']):
            lat = lats.isel(time=i)
            lon = lons.isel(time=i)
            fin = np.isfinite(lat + lon)
            if np.sum(fin) <= 3:
                area.append(0)
                continue
            lat = lat[fin]
            lon = lon[fin]
            aea = pyproj.Proj(
                f'+proj=aea +lat_0={lat.mean().values} +lat_1={lat.min().values} +lat_2={lat.max().values} +lon_0={lon.mean().values} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
            )

            x, y = aea(lat, lon, inverse=False)
            fin = np.isfinite(x + y)
            points = np.vstack((y.T, x.T)).T
            hull = ConvexHull(points)
            area.append(hull.volume)  # volume=area for 2D as here

        return np.array(area)
