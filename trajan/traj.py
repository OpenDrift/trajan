from abc import abstractmethod
import pyproj
import numpy as np
import xarray as xr
import cf_xarray as _
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def __require_obsdim__(f):
    """
    This decorator is for methods of Traj that require a time or obs dimension to work.
    """

    def wrapper(*args, **kwargs):
        if args[0].obsdim is None:
            raise ValueError(f'{f} requires an obs or time dimension')
        return f(*args, **kwargs)

    return wrapper

def detect_tx_dim(ds):
    if 'lon' in ds:
        return ds.lon
    elif 'longitude' in ds:
        return ds.longitude
    elif 'x' in ds:
        return ds.x
    elif 'X' in ds:
        return ds.X
    else:
        raise ValueError("Could not determine x / lon variable")


class Traj:
    ds: xr.Dataset

    __gcrs__: pyproj.CRS

    obsdim: str
    """
    Name of the dimension along which observations are taken. Usually either `obs` or `time`.
    """
    timedim: str

    def __init__(self, ds, obsdim, timedim):
        self.ds = ds
        self.__gcrs__ = pyproj.CRS.from_epsg(4326)
        self.obsdim = obsdim
        self.timedim = timedim

    @property
    def tx(self):
        """
        Trajectory x coordinates (usually longitude).

        .. see-also:

            * `ref:tlon`
            * `ref:ty`
        """
        return detect_tx_dim(self.ds)

    @property
    def ty(self):
        """
        Trajectory y coordinates (usually latitude).

        .. see-also:

            * `ref:tlat`
            * `ref:tx`
        """
        if 'lat' in self.ds:
            return self.ds.lat
        elif 'latitude' in self.ds:
            return self.ds.latitude
        elif 'y' in self.ds:
            return self.ds.y
        elif 'Y' in self.ds:
            return self.ds.Y
        else:
            raise ValueError("Could not determine y / lat variable")

    @property
    def tlon(self):
        """
        Retrieve the trajectories in geographic coordinates (longitudes).
        """
        if self.crs.is_geographic:
            return self.tx
        else:
            if self.crs is None:
                return self.tx
            else:
                x, _ = self.transform(self.__gcrs__, self.tx, self.ty)
                X = self.tx.copy(deep=False,
                                 data=x)  # TODO: remove grid-mapping.
                return X

    @property
    def tlat(self):
        """
        Retrieve the trajectories in geographic coordinates (latitudes).
        """
        if self.crs.is_geographic:
            return self.ty
        else:
            if self.crs is None:
                return self.ty
            else:
                _, y = self.transform(self.__gcrs__, self.tx, self.ty)
                Y = self.ty.copy(deep=False,
                                 data=y)  # TODO: remove grid-mapping.
                return Y

    def transform(self, to_crs, x, y):
        """
        Transform coordinates in this datasets coordinate system to `to_crs` coordinate system.

        Args:

            to_crs: `pyproj.CRS`.

            x, y: Coordinates in `self` CRS.

        Returns:

            xn, yn: Coordinates in `to_crs`.
        """
        t = pyproj.Transformer.from_crs(self.crs, to_crs, always_xy=True)
        return t.transform(x, y)

    def itransform(self, from_crs, x, y):
        """
        Transform coordinates in `from_crs` coordinate system to this datasets coordinate system.

        Args:

            from_crs: `pyproj.CRS`.

            x, y: Coordinates in `from_crs` CRS.

        Returns:

            xn, yn: Coordinates in this datasets CRS.
        """
        t = pyproj.Transformer.from_crs(from_crs, self.crs, always_xy=True)
        return t.transform(x, y)

    @property
    def crs(self) -> pyproj.CRS:
        """
        Retrieve the Proj.4 CRS from the CF-defined grid-mapping in the dataset.
        """
        if len(self.ds.cf.grid_mapping_names) == 0:
            logger.debug(
                f'No grid-mapping specified, checking if coordinates are lon/lat..'
            )
            if self.tx.name == 'lon' or self.tx.name == 'longitude':
                # assume this is in latlon projection
                return self.__gcrs__
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
        Returns a new dataset with the CF-supported grid-mapping / projection set to `crs`.
        """

        # TODO: Ideally this would be handled by cf-xarray or rio-xarray.

        ds = self.ds.copy()

        if crs is None:
            logger.info(
                'Removing CRS information and defining trajactories as Cartesian / unprojected data.'
            )

            if 'grid_mapping' in self.ds.cf:
                gm = self.ds.cf['grid_mapping']
                ds = ds.drop(gm.name)

            for var in self.ds:
                if 'grid_mapping' in self.ds[var].attrs:
                    del ds[var].attrs['grid_mapping']

            if self.tx.name == 'lon' or self.tx.name == 'longitude':
                logger.warning(
                    f'Renaming geographic {self.tx.name, self.ty.name} coordinates to x, y..'
                )
                ds['x'] = ds[self.tx.name].copy().rename('x')
                ds['y'] = ds[self.ty.name].copy().rename('y')
                assert ds['x'].name == 'x'

                ds = ds.drop_vars([self.tx.name, self.ty.name])

        else:
            gm = crs.to_cf()

            # Create grid mapping variable
            v = xr.DataArray(name=gm['grid_mapping_name'])
            v.attrs = gm
            ds[v.name] = v
            ds[self.tx.name].attrs['grid_mapping'] = v.name
            ds[self.ty.name].attrs['grid_mapping'] = v.name

        return ds

    @abstractmethod
    def is_1d(self):
        pass

    @abstractmethod
    def is_2d(self):
        pass

    def assign_cf_attrs(self,
                        creator_name=None,
                        creator_email=None,
                        title=None,
                        summary=None,
                        **kwargs) -> xr.Dataset:
        """
        Return a new dataset with CF-standard and common attributes set.
        """
        ds = self.ds.copy(deep=True)

        ds['trajectory'] = ds['trajectory'].astype(str)
        ds['trajectory'].attrs = {
            'cf_role': 'trajectory_id',
            'long_name': 'trajectory name'
        }

        ds = ds.assign_attrs({
            'Conventions':
            'CF-1.10',
            'featureType':
            'trajectory',
            'geospatial_lat_min':
            np.nanmin(self.tlat),
            'geospatial_lat_max':
            np.nanmax(self.tlat),
            'geospatial_lon_min':
            np.nanmin(self.tlon),
            'geospatial_lon_max':
            np.nanmax(self.tlon),
            'time_coverage_start':
            pd.to_datetime(
                np.nanmin(ds['time'].values[
                    ds['time'].values != np.datetime64('NaT')])).isoformat(),
            'time_coverage_end':
            pd.to_datetime(
                np.nanmax(ds['time'].values[
                    ds['time'].values != np.datetime64('NaT')])).isoformat(),
        })

        if creator_name:
            ds = ds.assign_attrs(creator_name=creator_name)

        if creator_email:
            ds = ds.assign_attrs(creator_email=creator_email)

        if title:
            ds = ds.assign_attrs(title=title)

        if summary:
            ds = ds.assign_attrs(summary=summary)

        if kwargs is not None:
            ds = ds.assign_attrs(**kwargs)

        return ds

    def index_of_last(self):
        """Find index of last valid position along each trajectory"""
        return np.ma.notmasked_edges(np.ma.masked_invalid(self.ds.lon.values),
                                     axis=1)[1][1]

    def speed(self):
        """Returns the speed [m/s] along trajectories"""

        distance = self.distance_to_next()
        timedelta_seconds = self.time_to_next() / np.timedelta64(1, 's')

        return distance / timedelta_seconds

    def distance_to(self, other):
        """
        Distance between trajectories or a single point.
        """

        other = other.broadcast_like(self.ds)
        geod = pyproj.Geod(ellps='WGS84')
        az_fwd, a2, distance = geod.inv(self.ds.traj.tlon, self.ds.traj.tlat,
                                        other.traj.tlon, other.traj.tlat)

        ds = xr.Dataset()
        ds['distance'] = xr.DataArray(distance,
                                      name='distance',
                                      coords=self.ds.traj.tlon.coords,
                                      attrs={'unit': 'm'})

        ds['az_fwd'] = xr.DataArray(az_fwd,
                                    name='forward azimuth',
                                    coords=self.ds.traj.tlon.coords,
                                    attrs={'unit': 'degrees'})

        ds['az_bwd'] = xr.DataArray(a2,
                                    name='back azimuth',
                                    coords=self.ds.traj.tlon.coords,
                                    attrs={'unit': 'degrees'})

        return ds

    @__require_obsdim__
    def distance_to_next(self):
        """Returns distance in m from one position to the next

           Last time is repeated for last position (which has no next position)
        """

        lon = self.ds.lon
        lat = self.ds.lat
        lenobs = self.ds.sizes[self.obsdim]
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

    @__require_obsdim__
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

    def convex_hull(self):
        """Returns the scipy convex hull for all particles, in geographical coordinates"""

        from scipy.spatial import ConvexHull

        lon = self.ds.lon.where(self.ds.status == 0)
        lat = self.ds.lat.where(self.ds.status == 0)
        fin = np.isfinite(lat + lon)
        if np.sum(fin) <= 3:
            return None
        if len(np.unique(lat)) == 1 and len(np.unique(lon)) == 1:
            return None
        lat = lat[fin]
        lon = lon[fin]
        points = np.vstack((lon.T, lat.T)).T
        return ConvexHull(points)

    def convex_hull_contains_point(self, lon, lat):
        """Returns True if given point is within the scipy convex hull for all particles"""
        from matplotlib.patches import Polygon

        hull = self.ds.traj.convex_hull()
        if hull is None:
            return False
        p = Polygon(hull.points[hull.vertices])
        point = np.c_[lon, lat]
        return p.contains_points(point)[0]

    def get_area_convex_hull(self):
        """Returns the area [m2] of the convex hull spanned by all particles"""

        from scipy.spatial import ConvexHull

        lon = self.ds.lon.where(self.ds.status == 0)
        lat = self.ds.lat.where(self.ds.status == 0)
        fin = np.isfinite(lat + lon)
        if np.sum(fin) <= 3:
            return 0
        if len(np.unique(lat)) == 1 and len(np.unique(lon)) == 1:
            return 0
        lat = lat[fin]
        lon = lon[fin]
        # An equal area projection centered around the particles
        aea = pyproj.Proj(
            f'+proj=aea +lat_0={lat.mean().values} +lat_1={lat.min().values} +lat_2={lat.max().values} +lon_0={lon.mean().values} +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
        )

        x, y = aea(lat, lon, inverse=False)
        fin = np.isfinite(x + y)
        points = np.vstack((y.T, x.T)).T
        hull = ConvexHull(points)
        return np.array(hull.volume)  # volume=area for 2D as here

    @abstractmethod
    def gridtime(self, times, timedim = None):
        """Interpolate dataset to a regular time interval or a different grid.

        Args:

            `times`: Target time interval, can be either:
                - an array of times, or
                - a string "freq" specifying a Pandas daterange (e.g. 'h', '6h, 'D'...) suitable for `pd.date_range`.

            `timedime`: Name of new time dimension. The default is to use the same name as previously.

        Returns:

            A new dataset interpolated to the target times. The dataset will be 1D (i.e. gridded) and the time dimension will be named `time`.
        """
