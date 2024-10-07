"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

from abc import abstractmethod
import pyproj
import numpy as np
import xarray as xr
import cf_xarray as _
import pandas as pd
import logging

from .plot import Plot
from .animation import Animation

logger = logging.getLogger(__name__)

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


def detect_time_dim(ds, obsdim):
    logger.debug(f'Detecting time-dimension for "{obsdim}"..')
    for v in ds.variables:
        if obsdim in ds[v].dims and 'time' in v:
            return v

    raise ValueError("no time dimension detected")


class Traj:
    ds: xr.Dataset

    __plot__: Plot
    __animate__: Animation

    __gcrs__: pyproj.CRS

    obsdim: str
    """
    Name of the dimension along which observations are taken. Usually either `obs` or `time`.
    """
    timedim: str

    def __init__(self, ds, obsdim, timedim):
        self.ds = ds
        self.__plot__ = None
        self.__animate__ = None
        self.__gcrs__ = pyproj.CRS.from_epsg(4326)
        self.obsdim = obsdim
        self.timedim = timedim


    @property
    def plot(self) -> Plot:
        """
        See :class:`trajan.plot.Plot`.
        """
        if self.__plot__ is None:
            logger.debug(f'Setting up new plot object.')
            self.__plot__ = Plot(self.ds)

        return self.__plot__

    @property
    def animate(self):
        """
        See :class:`trajan.animation.Animation`.
        """
        if self.__animate__ is None:
            logger.debug(f'Setting up new animation object.')
            self.__animate__ = Animation(self.ds)

        return self.__animate__

    @property
    def tx(self):
        """
        Trajectory x coordinates (usually longitude) - test.

        See Also
        --------
        tlon, ty
        """

        return detect_tx_dim(self.ds)

    @property
    def ty(self):
        """
        Trajectory y coordinates (usually latitude).

        See Also
        --------
        tlat, tx
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
    def tlat(self) -> xr.DataArray:
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

        Parameters
        ----------
        to_crs : pyproj.CRS

        x, y : arrays
            Coordinates in `self` CRS

        Returns
        -------
        xn, yn : arrays
            Coordinates in `to_crs`
        """
        t = pyproj.Transformer.from_crs(self.crs, to_crs, always_xy=True)
        return t.transform(x, y)

    def itransform(self, from_crs, x, y):
        """
        Transform coordinates in `from_crs` coordinate system to this datasets coordinate system.

        Parameters
        ----------
        from_crs : `pyproj.CRS`

        x, y : arrays
            Coordinates in `from_crs` CRS

        Returns
        -------
        xn, yn : arrays
            Coordinates in this datasets CRS
        """
        t = pyproj.Transformer.from_crs(from_crs, self.crs, always_xy=True)
        return t.transform(x, y)

    @property
    def crs(self) -> pyproj.CRS:
        """
        Retrieve the pyproj.CRS object from the CF-defined grid-mapping in the dataset.
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

    def set_crs(self, crs) -> xr.Dataset:
        """
        Returns a new dataset with the CF-supported grid-mapping / projection set to `crs`.

        Parameters
        ----------
        crs: pyproj.CRS

        Returns
        -------
        updated dataset

        .. warning::

            This does not transform the coordinates, make sure that `crs` is matching the data in the dataset.
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
        """Returns True if dataset is 1D, i.e. time is a 1D coordinate variable."""

    @abstractmethod
    def is_2d(self):
        """Returns True if dataset is 2D, i.e. time is a 2D variable and not a coordinate variable."""

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
        """Find index of last valid position along each trajectory."""
        return np.ma.notmasked_edges(np.ma.masked_invalid(self.ds.lon.values),
                                     axis=1)[1][1]

    @abstractmethod
    def speed(self) -> xr.DataArray:
        """Returns the speed [m/s] along trajectories.

        Calculates the speed by dividing the distance between two points
        along trajectory by the corresponding time step.

        Returns
        -------
        xarray.DataArray
            Same dimensions as original dataset, since last value is repeated along time dimension

        See Also
        --------
        distance_to_next, time_to_next

        """
        distance = self.distance_to_next()
        timedelta_seconds = self.time_to_next() / np.timedelta64(1, 's')

        return distance / timedelta_seconds

    @abstractmethod
    def time_to_next(self) -> pd.Timedelta:
        """Returns the timedelta between time steps.

        Returns
        -------
        xarray.DataArray
            Scalar timedelta for 1D objects, and DataArray of same size as input for 2D objects         

        See Also
        --------
        distance_to_next, speed
        """
        pass

    @abstractmethod
    def velocity_spectrum(self) -> xr.DataArray:
        pass

    # def rotary_spectrum(self):
    #     pass

    def distance_to(self, other) -> xr.Dataset:
        """
        Distance between trajectories or a single point.

        Parameters
        ----------
        other : xarray.DataSet
            Other dataset to which distance is calculated

        Returns
        -------
        xarray.Dataset
            Same dimensions as original dataset

        See Also
        --------
        distance_to_next

        """

        other = other.broadcast_like(self.ds)
        geod = pyproj.Geod(ellps='WGS84')
        az_fwd, a2, distance = geod.inv(self.ds.traj.tlon, self.ds.traj.tlat,
                                        other.traj.tlon, other.traj.tlat)

        ds = xr.Dataset()
        ds['distance'] = xr.DataArray(distance,
                                      name='distance',
                                      coords=self.ds.traj.tlon.coords,
                                      attrs={'units': 'm'})

        ds['az_fwd'] = xr.DataArray(az_fwd,
                                    name='forward azimuth',
                                    coords=self.ds.traj.tlon.coords,
                                    attrs={'units': 'degrees'})

        ds['az_bwd'] = xr.DataArray(a2,
                                    name='back azimuth',
                                    coords=self.ds.traj.tlon.coords,
                                    attrs={'units': 'degrees'})

        return ds

    def distance_to_next(self):
        """Returns distance in m from one position to the next.

        Last time is repeated for last position (which has no next position)

        Returns
        -------
        xarray.DataArray
            Same dimensions as original dataset, since last value is repeated along time dimension

        See Also
        --------
        time_to_next, speed

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

    def azimuth_to_next(self):
        """Returns azimution travel direction in degrees from one position to the next.

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
        """Returns velocity components [m/s] from one position to the next.

           Last time is repeated for last position (which has no next position)
        """
        speed = self.speed()
        azimuth = self.azimuth_to_next()

        u = speed * np.cos(azimuth)
        v = speed * np.sin(azimuth)

        return u, v

    def convex_hull(self):
        """Return the scipy convex hull for all particles, in geographical coordinates.

        Returns
        -------
        scipy.spatial.ConvexHull
            Convex Hull around all positions of given Dataset.
        """

        from scipy.spatial import ConvexHull

        lon = self.ds.lon
        lat = self.ds.lat
        if 'status' in self.ds.variables:
            lon = lon.where(self.ds.status == 0)
            lat = lat.where(self.ds.status == 0)
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
        """Return True if given point is within the scipy convex hull for all particles.

        Parameters
        ----------
        lon, lat    scalars
            longitude and latitude [degrees] of a position.

        Returns
        -------
        bool
            True if convex hull of positions in cataset contains given position.
        """
        from matplotlib.patches import Polygon

        hull = self.ds.traj.convex_hull()
        if hull is None:
            return False
        p = Polygon(hull.points[hull.vertices])
        point = np.c_[lon, lat]
        return p.contains_points(point)[0]

    def get_area_convex_hull(self):
        """Return the area [m2] of the convex hull spanned by all particles."""

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
    def gridtime(self, times, timedim = None) -> xr.Dataset:
        """Interpolate dataset to a regular time interval or a different grid.

        Parameters
        ----------
        times : array or str
            Target time interval, can be either:
                - an array of times, or
                - a string specifying a Pandas daterange (e.g. 'h', '6h, 'D'...) suitable for `pd.date_range`.

        timedim : str
            Name of new time dimension. The default is to use the same name as previously.

        Returns
        -------
        Dataset
            A new dataset interpolated to the target times. The dataset will be 1D (i.e. gridded) and the time dimension will be named `time`.
        """

    @abstractmethod
    def seltime(self, t0=None, t1=None) -> xr.Dataset:
        """ Select observations in time window between `t0` and `t1` (inclusive). """


    @abstractmethod
    def skill(self, other, method='liu-weissberg', **kwargs) -> xr.Dataset:
        """
        Compare the skill score between this trajectory and `other`.

        Parameters
        ----------

        other : Dataset
            Another trajectory dataset.

        method : str
            skill-score method, currently only liu-weissberg.

        **kwargs :
            Passed on to the skill-score method.

        Returns
        -------

        skill : :class:`xarray.Dataset`
            The skill-score in the same dimensions as this dataset.

        Notes
        -----
        The datasets must be sampled (or have observations) at approximately the same timesteps. Consider using :meth:`gridtime` to interpolate one of the datasets to the other.

        The datasets must have the same number of trajectories. If you wish to compare a single trajectory to many others, duplicate it along the trajectory dimension to match the trajectory dimension of the other. See further down for an example.

        Examples
        --------

        >>> import xarray as xr
        >>> import trajan as _
        >>> import lzma
        >>> b = lzma.open('examples/barents.nc.xz')
        >>> ds = xr.open_dataset(b)
        >>> other = ds.copy()
        >>> ds = ds.traj.gridtime('1h')
        >>> other = other.traj.gridtime(ds.time)
        >>> skill = ds.traj.skill(other)

        >>> skill
        <xarray.DataArray 'Skillscore' (trajectory: 2)> Size: 8B
        array([1., 1.], dtype=float32)
        Coordinates:
          * trajectory  (trajectory) int64 16B 0 1
        Attributes:
            method:   liu-weissberg

        If you need to broadcast a dataset with a single drifter to one with many you can use `xarray.broadcast` or `xarray.Dataset.broadcast_like`:

        .. note::

            If the other dataset has any other dimensions, on any other variables, you need to exclude those when broadcasting.

        >>> b0 = ds.isel(trajectory=0) # `b0` now only has a single drifter (no trajectory dimension)
        >>> b0 = b0.broadcast_like(ds)
        >>> skill = b0.traj.skill(ds)

        >>> skill
        <xarray.DataArray 'Skillscore' (trajectory: 2)> Size: 8B
        array([1.      , 0.608058], dtype=float32)
        Coordinates:
          * trajectory  (trajectory) int64 16B 0 1
        Attributes:
            method:   liu-weissberg
        """
