import pyproj
import numpy as np
import xarray as xr
import cf_xarray as _
import logging

logger = logging.getLogger(__name__)


class Traj:
    ds: xr.Dataset
    __gcrs__: pyproj.CRS
    obsdim: str

    def __init__(self, ds):
        self.ds = ds
        self.__gcrs__ = pyproj.CRS.from_epsg(4326)

        if 'obs' in self.ds.dims:
            self.obsdim = 'obs'
        elif 'time' in self.ds.dims:
            self.obsdim = 'time'
        else:
            raise ValueError('No time or obs dimension')

    @property
    def tx(self):
        """
        Trajectory x coordinates (usually longitude).

        .. see-also:

            `ref:tlat`
        """
        if 'lon' in self.ds:
            return self.ds.lon
        elif 'longitude' in self.ds:
            return self.ds.longitude
        elif 'x' in self.ds:
            return self.ds.x
        elif 'X' in self.ds:
            return self.ds.X
        else:
            raise ValueError("Could not determine x / lon variable")

    @property
    def ty(self):
        """
        Trajectory y coordinates (usually latitude).

        .. see-also:

            `ref:tlon`
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
            if len(np.unique(lat)) == 1 and len(np.unique(lon)) == 1:
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

    def gridtime(self, times):
        """Interpolate dataset to regular time interval

        times:
            - an array of times, or
            - a string "freq" specifying a Pandas daterange (e.g. 'h', '6h, 'D'...)

        Note that the resulting DataSet will have "time" as a dimension coordinate.
        """

        from scipy.interpolate import interp1d
        import pandas as pd

        if isinstance(times, str):  # Make time series with given interval
            freq = times
            start_time = np.nanmin(np.asarray(self.ds.time))
            # start_time = pd.to_datetime(start_time).strftime('%Y-%m-%d')
            end_time = np.nanmax(np.asarray(self.ds.time))
            # end_time = pd.to_datetime(end_time).strftime('%Y-%m-%d')
            times = pd.date_range(start_time, end_time, freq=freq, inclusive='both')

        # Why not..
        if not isinstance(times, np.ndarray):
            times = times.to_numpy()

        # Create empty dataset to hold interpolated values
        d = xr.Dataset(coords={
            'trajectory': self.ds['trajectory'],
            'time': (["time"], times)
        },
                       attrs=self.ds.attrs)

        for varname, var in self.ds.variables.items():
            if varname in ('time', 'obs'):
                continue
            if 'obs' not in var.dims:
                d[varname] = var
                continue

            # Create empty dataarray to hold interpolated values for given variable
            da = xr.DataArray(data=np.zeros(
                tuple(d.dims[di] for di in ['trajectory', 'time'])) * np.nan,
                              dims=d.dims,
                              coords=d.coords,
                              attrs=var.attrs)

            origtimes = self.ds['time'].ffill(dim='obs').astype(np.float64)

            for t in range(
                    self.ds.dims['trajectory']):  # loop over trajectories
                # Make interpolator
                f = interp1d(origtimes.isel(trajectory=t), var.isel(trajectory=t), bounds_error=False)
                # Interpolate onto given times
                da.loc[{'trajectory': t}] = f(times.astype(np.float64))

            d[varname] = da

        return d

