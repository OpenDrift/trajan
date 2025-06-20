"""
Extending xarray Dataset with functionality specific to trajectory datasets.

Presently supporting Cf convention H.4.1
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories.
"""

from abc import abstractmethod
from datetime import timedelta
from functools import cache
import pyproj
import numpy as np
import xarray as xr
import cf_xarray as _
import pandas as pd
import logging
import cartopy.crs

from .plot import Plot
from .animation import Animation

logger = logging.getLogger(__name__)


def detect_tx_variable(ds):
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


def ensure_time_dim(ds, time_dim):
    if not time_dim in ds.dims:
        return ds.expand_dims(time_dim)
    else:
        return ds

def grid_area(lons, lats):
    """Calculate the area of each grid cell"""
    from shapely.geometry import Polygon

    if lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    geod = pyproj.Geod(ellps="WGS84")
    rows, cols = lons.shape
    grid_areas = np.zeros((rows - 1, cols - 1))

    for i in range(rows - 1):
        for j in range(cols - 1):
            lon = [lons[i, j], lons[i, j + 1], lons[i + 1, j + 1], lons[i + 1, j]]
            lat = [lats[i, j], lats[i, j + 1], lats[i + 1, j + 1], lats[i + 1, j]]
            polygon = Polygon([(lon[0], lat[0]), (lon[1], lat[1]), (lon[2], lat[2]), (lon[3], lat[3])])
            grid_areas[i, j] = abs(geod.geometry_area_perimeter(polygon)[0])

    return grid_areas


class Traj:
    ds: xr.Dataset

    __plot__: Plot
    __animate__: Animation

    __gcrs__: pyproj.CRS

    def __init__(self, ds, trajectory_dim, obs_dim, time_varname):
        self.ds = ds
        self.__plot__ = None
        self.__animate__ = None
        self.__gcrs__ = pyproj.CRS.from_epsg(4326)
        self.trajectory_dim = trajectory_dim  # name of trajectory dimension
        self.obs_dim = obs_dim  # dimension along which time increases
        self.time_varname = time_varname

    def __repr__(self):
        output = '=======================\n'
        output += 'TrajAn info:\n'
        output += '------------\n'
        if self.trajectory_dim is None:
            output += 'Single trajectory (no trajectory dimension)\n'
        else:
            output += f'{self.ds.sizes[self.trajectory_dim]} trajectories  [trajectory_dim: {self.trajectory_dim}]\n'
        if self.time_varname is not None:
            output += f'{self.ds.sizes[self.obs_dim]} timesteps      [obs_dim: {self.obs_dim}]\n'
            timevar = self.ds[self.time_varname]
            output += f'Time variable:    {timevar.name}{list(timevar.sizes)}   ({len(timevar.sizes)}D)\n'
            try:
                timestep = self.timestep()
                timestep = timedelta(seconds=int(timestep))
            except:
                timestep = '[self.timestep returns error]'  # TODO
            output += f'Timestep:       {timestep}\n'
            start_time = self.ds.time.min(skipna=True).data
            end_time = self.ds.time.max(skipna=True).data
            output += f'Time coverage:  {start_time} - {end_time}\n'
        else:
            output += f'Dataset has no time variable'
        output += f'Longitude span: {self.tx.min(skipna=True).data} to {self.tx.max(skipna=True).data}\n'
        output += f'Latitude span:  {self.ty.min(skipna=True).data} to {self.ty.max(skipna=True).data}\n'
        output += 'Variables:\n'
        for var in self.ds.variables:
            if var not in ['trajectory', self.obs_dim]:
                output += f'    {var}'
                if 'standard_name' in self.ds[var].attrs:
                    output += f'  [{self.ds[var].standard_name}]'
                output += '\n'
        output += '=======================\n'
        return output

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
        Trajectory x coordinates (usually longitude).

        See Also
        --------
        tlon, ty
        """

        return detect_tx_variable(self.ds)

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

        See Also
        --------
        tx, tlat
        """
        if self.crs.is_geographic:
            return self.tx
        else:
            if self.crs is None:
                return self.tx
            else:
                return self.transform(self.__gcrs__).traj.tlon

    @property
    def tlat(self) -> xr.DataArray:
        """
        Retrieve the trajectories in geographic coordinates (latitudes).

        See Also
        --------
        ty, tlon
        """
        if self.crs.is_geographic:
            return self.ty
        else:
            if self.crs is None:
                return self.ty
            else:
                return self.transform(self.__gcrs__).traj.tlat

    @cache
    def transform(self, to_crs):
        """
        Transform this datasets to `to_crs` coordinate system. If the target
        projection is not a geographic coordinate system the variables will be
        named `x` and `y`, otherwise `lon` and `lat`.

        Parameters
        ----------
        to_crs : pyproj.crs.CRS


        Returns
        -------
        ds : array-like
            Dataset in `to_crs` coordinate system.

        Example
        -------

        Transform dataset to UTM coordinates:

        >>> import xarray as xr
        >>> import trajan as _
        >>> import lzma
        >>> import pyproj
        >>> b = lzma.open('examples/barents.nc.xz')
        >>> ds = xr.open_dataset(b)
        >>> print(ds)
        <xarray.Dataset> Size: 110kB
        Dimensions:        (trajectory: 2, obs: 2287)
        Dimensions without coordinates: trajectory, obs
        Data variables:
            lon            (trajectory, obs) float64 37kB ...
            lat            (trajectory, obs) float64 37kB ...
            time           (trajectory, obs) datetime64[ns] 37kB ...
            drifter_names  (trajectory) <U16 128B ...
        Attributes: (12/13)
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   74.5454462
            geospatial_lat_max:   77.4774768
            geospatial_lon_min:   17.2058074
            geospatial_lon_max:   29.8523485
            ...                   ...
            time_coverage_end:    2022-11-23T13:30:28
            creator_email:        gauteh@met.no, knutfd@met.no
            creator_name:         Gaute Hope and Knut Frode Dagestad
            creator_url:          https://github.com/OpenDrift/opendrift
            summary:              Two drifters in the Barents Sea. One stranded at Ho...
            title:                Barents Sea drifters

        >>> crs = pyproj.CRS.from_epsg(3857) # mercator
        >>>
        >>> ds_merc = ds.traj.transform(crs)
        >>> print(ds_merc)
        <xarray.Dataset> Size: 110kB
        Dimensions:        (trajectory: 2, obs: 2287)
        Dimensions without coordinates: trajectory, obs
        Data variables:
            time           (trajectory, obs) datetime64[ns] 37kB ...
            drifter_names  (trajectory) <U16 128B ...
            x              (trajectory, obs) float64 37kB 3.323e+06 ... 2.354e+06
            y              (trajectory, obs) float64 37kB 1.401e+07 ... 1.276e+07
            proj1          float64 8B nan
        Attributes: (12/13)
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   74.5454462
            geospatial_lat_max:   77.4774768
            geospatial_lon_min:   17.2058074
            geospatial_lon_max:   29.8523485
            ...                   ...
            time_coverage_end:    2022-11-23T13:30:28
            creator_email:        gauteh@met.no, knutfd@met.no
            creator_name:         Gaute Hope and Knut Frode Dagestad
            creator_url:          https://github.com/OpenDrift/opendrift
            summary:              Two drifters in the Barents Sea. One stranded at Ho...
            title:                Barents Sea drifters


        See also
        --------
        transformer, crs
        """
        t = pyproj.Transformer.from_crs(self.crs, to_crs, always_xy=True)
        tx, ty = t.transform(self.tx, self.ty)

        xvar = self.tx.name
        yvar = self.ty.name

        ds = self.ds.copy().drop_vars([xvar, yvar])

        tx = xr.DataArray(tx, coords=self.tx.coords)
        ty = xr.DataArray(ty, coords=self.ty.coords)

        if to_crs.is_geographic:
            # TODO: may exist..
            ds['lon'] = tx
            ds['lat'] = ty
        else:
            ds['x'] = tx
            ds['y'] = ty

        ds = ds.traj.set_crs(to_crs)

        return ds

    def transformer(self, from_crs):
        """
        Create a transformer useful for transforming other coordinates to the CRS of this dataset.

        Parameters
        ---------
        from_crs : pyproj.crs.CRS

        Returns
        -------
        transformer: pyproj.Transformer


        Example
        -------

        Transform UTM coordinates to lat-lon

        >>> import xarray as xr
        >>> import trajan as _
        >>> import lzma
        >>> import pyproj
        >>> b = lzma.open('examples/barents.nc.xz')
        >>> ds = xr.open_dataset(b)
        >>> print(ds)
        <xarray.Dataset> Size: 110kB
        Dimensions:        (trajectory: 2, obs: 2287)
        Dimensions without coordinates: trajectory, obs
        Data variables:
            lon            (trajectory, obs) float64 37kB ...
            lat            (trajectory, obs) float64 37kB ...
            time           (trajectory, obs) datetime64[ns] 37kB ...
            drifter_names  (trajectory) <U16 128B ...
        Attributes: (12/13)
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   74.5454462
            geospatial_lat_max:   77.4774768
            geospatial_lon_min:   17.2058074
            geospatial_lon_max:   29.8523485
            ...                   ...
            time_coverage_end:    2022-11-23T13:30:28
            creator_email:        gauteh@met.no, knutfd@met.no
            creator_name:         Gaute Hope and Knut Frode Dagestad
            creator_url:          https://github.com/OpenDrift/opendrift
            summary:              Two drifters in the Barents Sea. One stranded at Ho...
            title:                Barents Sea drifters

        >>> crs = pyproj.CRS.from_epsg(3857) # mercator
        >>>
        >>> tlon, tlat = ds.traj.transformer(crs).transform(-10000, 3000)
        >>> print(tlon, tlat)
        -0.08983152841195213 0.026949457529889528

        See also
        --------
        transform, crs
        """
        return pyproj.Transformer.from_crs(from_crs, self.crs, always_xy=True)

    @property
    def crs(self) -> pyproj.crs.CRS:
        """
        Retrieve the pyproj.crs.CRS object from the CF-defined grid-mapping in the dataset.

        Returns
        -------
        pyproj.crs.CRS

        See also
        --------
        transform, set_crs
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
            return pyproj.crs.CRS.from_cf(gm.attrs)

    @property
    def ccrs(self) -> cartopy.crs.CRS:
        """
        Retrieve a cartopy CRS from the pyproj CRS.

        Returns
        -------
        cartopy.crs.CRS

        Warning
        -------
        This may not be totally accurate.

        See also
        --------
        crs, transform
        """

        crs = self.crs
        if crs is not None:
            return cartopy.crs.CRS(crs.to_json_dict())
        else:
            return None

    def set_crs(self, crs) -> xr.Dataset:
        """
        Returns a new dataset with the CF-supported grid-mapping / projection set to `crs`.

        Parameters
        ----------
        crs : pyproj.crs.CRS

        Returns
        -------
        Dataset
            Updated dataset

        Warning
        -------
        You most likely want `transform`. This does not transform the coordinates, make sure that `crs` is matching the data in the dataset.

        See also
        --------
        transform
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

            name = gm.get('grid_mapping_name', 'proj1')

            # Create grid mapping variable
            v = xr.DataArray(name=name)
            v.attrs = gm
            ds[v.name] = v
            ds[self.tx.name].attrs['grid_mapping'] = v.name
            ds[self.ty.name].attrs['grid_mapping'] = v.name

        return ds

    @abstractmethod
    def is_1d(self) -> bool:
        """Returns True if dataset is 1D, i.e. time is a 1D coordinate variable.

        Returns
        -------
        bool
        """

    @abstractmethod
    def is_2d(self) -> bool:
        """Returns True if dataset is 2D, i.e. time is a 2D variable and not a coordinate variable.

        Returns
        -------
        bool
        """

    def assign_cf_attrs(self,
                        creator_name=None,
                        creator_email=None,
                        title=None,
                        summary=None,
                        **kwargs) -> xr.Dataset:
        """
        Return a new dataset with CF-standard and common attributes set.

        Parameters
        ----------
        creator_name : string
            Creator of dataset (optional).

        creator_email : string
            Creator email (optional).

        title : string
            Title of dataset (optional).

        summary : string
            Description of dataset (optional).

        *kwargs : dict
            Additional attribute key and values (optional).

        Returns
        -------
        Dataset
            Updated dataset with provided attributes, in addition to several CF standard attributes,
            including Conventions, featureType, geospatial_lat_min etc.


        """
        ds = self.ds.copy(deep=True)

        ds[self.trajectory_dim] = ds[self.trajectory_dim].astype(str)
        ds[self.trajectory_dim].attrs = {
            'cf_role': 'trajectory_id',
            'long_name': 'trajectory name'
        }

        ds = ds.assign_attrs({
            'Conventions':
            'CF-1.10',
            'featureType':
            'trajectory',
            'geospatial_lat_min':
            np.nanmin(self.tlat)
            if self.ds.sizes[self.obs_dim] > 0 else np.nan,
            'geospatial_lat_max':
            np.nanmax(self.tlat)
            if self.ds.sizes[self.obs_dim] > 0 else np.nan,
            'geospatial_lon_min':
            np.nanmin(self.tlon)
            if self.ds.sizes[self.obs_dim] > 0 else np.nan,
            'geospatial_lon_max':
            np.nanmax(self.tlon)
            if self.ds.sizes[self.obs_dim] > 0 else np.nan,
            'time_coverage_start':
            pd.to_datetime(
                np.nanmin(ds[self.time_varname].values[ds[
                    self.time_varname].values != np.datetime64('NaT')])).
            isoformat() if self.ds.sizes[self.obs_dim] > 0 else np.nan,
            'time_coverage_end':
            pd.to_datetime(
                np.nanmax(ds[self.time_varname].values[ds[
                    self.time_varname].values != np.datetime64('NaT')])
            ).isoformat() if self.ds.sizes[self.obs_dim] > 0 else np.nan,
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
        """Find index of last valid position along each trajectory.

        Returns
        -------
        array-like
            Array of the index of the last valid position along each trajectory.
        """
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
        DataArray
            Scalar timedelta for 1D objects (fixed timestep), and DataArray of same size as input for 2D objects

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

    @abstractmethod
    def distance_to(self, other) -> xr.Dataset:
        """
        Distance between trajectories or a single point.

        Parameters
        ----------
        other : Dataset
            Other dataset to which distance is calculated

        Returns
        -------
        Dataset
            Same dimensions as original dataset, containing three DataArrays from pyproj.geod.inv:
              distance : distance [meters]

              az_fwd : forward azimuth angle [degrees]

              az_bwd : backward azimuth angle [degrees]

        See Also
        --------
        distance_to_next

        """
        pass

    def length(self):
        """Returns distance in meters of each trajectory.

        Returns
        -------
        DataArray
            With trajectories and lengths.

        See Also
        --------
        distance_to_next
        """

        # TODO: remove isel if distance_to_next doesn't return last
        # distance twice.
        l = self.distance_to_next().isel({
            self.obs_dim: slice(0, -1)
        }).sum(dim=self.obs_dim, skipna=True)
        l.name = 'length'
        l = l.assign_attrs({
            'unit': 'm',
            'description': 'length of trajectory'
        })

        return l

    def distance_to_next(self):
        """Returns distance in meters from one position to the next along trajectories.

        Last time is repeated for last position (which has no next position).

        Returns
        -------
        DataArray
            Same dimensions as original Dataset, since last value is repeated along time dimension.

        See Also
        --------
        azimuth_to_next, time_to_next, speed

        """

        lon = self.ds.lon
        lat = self.ds.lat
        lenobs = self.ds.sizes[self.obs_dim]
        lonfrom = lon.isel({self.obs_dim: slice(0, lenobs - 1)})
        latfrom = lat.isel({self.obs_dim: slice(0, lenobs - 1)})
        lonto = lon.isel({self.obs_dim: slice(1, lenobs)})
        latto = lat.isel({self.obs_dim: slice(1, lenobs)})
        geod = pyproj.Geod(ellps='WGS84')
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto,
                                                 latto)

        distance = xr.DataArray(distance, coords=lonfrom.coords, dims=lon.dims)
        distance = xr.concat((distance, distance.isel({self.obs_dim: -1})),
                             dim=self.obs_dim)  # repeating last time step to
        return distance

    def azimuth_to_next(self):
        """Returns azimution travel direction in degrees from one position to the next.

        Last time is repeated for last position (which has no next position).

        Returns
        -------
        DataArray
            Same dimensions as original Dataset, since last value is repeated along time dimension.

        See Also
        --------
        distance_to_next, time_to_next, speed
        """

        # TODO: method is almost duplicate of "distance_to_next" above
        lon = self.ds.lon
        lat = self.ds.lat
        lenobs = self.ds.dims[self.obs_dim]
        lonfrom = lon.isel({self.obs_dim: slice(0, lenobs - 1)})
        latfrom = lat.isel({self.obs_dim: slice(0, lenobs - 1)})
        lonto = lon.isel({self.obs_dim: slice(1, lenobs)})
        latto = lat.isel({self.obs_dim: slice(1, lenobs)})
        geod = pyproj.Geod(ellps='WGS84')
        azimuth_forward, a2, distance = geod.inv(lonfrom, latfrom, lonto,
                                                 latto)

        azimuth_forward = xr.DataArray(azimuth_forward,
                                       coords=lonfrom.coords,
                                       dims=lon.dims)
        azimuth_forward = xr.concat(
            (azimuth_forward, azimuth_forward.isel({self.obs_dim: -1})),
            dim=self.obs_dim)  # repeating last time step to
        return azimuth_forward

    def velocity_components(self):
        """Returns velocity components [m/s] from one position to the next.

        Last time is repeated for last position (which has no next position)

        Returns
        -------
        (u, v) : array_like
            East and north components of velocities at given position along trajectories.
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
        lon, lat  :  scalar
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
        """Return the area [m2] of the convex hull spanned by all positions.

        Returns
        -------
        scalar
            Area [m2] of convex hull around all positions.
        """

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
    def gridtime(self, times, time_varname=None) -> xr.Dataset:
        """Interpolate dataset to a regular time interval or a different grid.

        Parameters
        ----------
        times : array or str
            Target time interval, can be either:
                - an array of times, or
                - a string specifying a Pandas daterange (e.g. 'h', '6h, 'D'...) suitable for `pd.date_range`.

        time_varname : str
            Name of new time dimension. The default is to use the same name as previously.

        Returns
        -------
        Dataset
            A new dataset interpolated to the target times. The dataset will be 1D (i.e. gridded) and the time dimension will be named `time`.
        """

    @abstractmethod
    def sel(self, *args, **kwargs) -> xr.Dataset:
        """Select on each trajectory. On 1D datasets this is just a shortcut for `Dataset.sel`.

        Parameters
        ----------
        Anything accepted by `Dataset.sel`.

        Returns
        -------

        ds : Dataset
            A dataset with the selected range in each trajectory.

        See also
        --------
        iseltime, sel, isel
        """

    @abstractmethod
    def seltime(self, t0=None, t1=None) -> xr.Dataset:
        """Select observations in time window between `t0` and `t1` (inclusive). For 1D datasets prefer to use `xarray.Dataset.sel`.

        Parameters
        ----------
        t0, t1 : numpy.datetime64

        Returns
        -------

        ds : Dataset
            A dataset with the selected indexes in each trajectory.

        See also
        --------
        iseltime, sel, isel
        """

    @abstractmethod
    def iseltime(self, i) -> xr.Dataset:
        """Select observations by index (of non-nan, time, observation) across
        trajectories. For 1D datasets prefer to use `xarray.Dataset.isel`.

        Parameters
        ----------
        i : index, list of indexes or a slice.


        Returns
        -------

        ds : Dataset
            A dataset with the selected indexes in each trajectory.


        Example
        -------

        Select the first and last element in each trajectory in a dataset of
        unstructured observations:

        >>> import xarray as xr
        >>> import trajan as _
        >>> import lzma
        >>> b = lzma.open('examples/barents.nc.xz')
        >>> ds = xr.open_dataset(b)
        >>> print(ds)
        <xarray.Dataset> Size: 110kB
        Dimensions:        (trajectory: 2, obs: 2287)
        Dimensions without coordinates: trajectory, obs
        Data variables:
            lon            (trajectory, obs) float64 37kB ...
            lat            (trajectory, obs) float64 37kB ...
            time           (trajectory, obs) datetime64[ns] 37kB ...
            drifter_names  (trajectory) <U16 128B ...
        Attributes: (12/13)
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   74.5454462
            geospatial_lat_max:   77.4774768
            geospatial_lon_min:   17.2058074
            geospatial_lon_max:   29.8523485
            ...                   ...
            time_coverage_end:    2022-11-23T13:30:28
            creator_email:        gauteh@met.no, knutfd@met.no
            creator_name:         Gaute Hope and Knut Frode Dagestad
            creator_url:          https://github.com/OpenDrift/opendrift
            summary:              Two drifters in the Barents Sea. One stranded at Ho...
            title:                Barents Sea drifters

        >>> ds = ds.traj.iseltime([0, -1])
        >>> print(ds)
        <xarray.Dataset> Size: 224B
        Dimensions:        (trajectory: 2, obs: 2)
        Dimensions without coordinates: trajectory, obs
        Data variables:
            lon            (trajectory, obs) float64 32B 29.85 25.11 27.82 21.15
            lat            (trajectory, obs) float64 32B 77.3 76.57 77.11 74.58
            time           (trajectory, obs) datetime64[ns] 32B 2022-10-07T00:00:38 ....
            drifter_names  (trajectory) <U16 128B 'UIB-2022-TILL-01' 'UIB-2022-TILL-02'
        Attributes: (12/13)
            Conventions:          CF-1.10
            featureType:          trajectory
            geospatial_lat_min:   74.5454462
            geospatial_lat_max:   77.4774768
            geospatial_lon_min:   17.2058074
            geospatial_lon_max:   29.8523485
            ...                   ...
            time_coverage_end:    2022-11-23T13:30:28
            creator_email:        gauteh@met.no, knutfd@met.no
            creator_name:         Gaute Hope and Knut Frode Dagestad
            creator_url:          https://github.com/OpenDrift/opendrift
            summary:              Two drifters in the Barents Sea. One stranded at Ho...
            title:                Barents Sea drifters

        See also
        --------
        seltime, sel, isel
        """

    @abstractmethod
    def skill(self, expected, method='liu-weissberg', **kwargs) -> xr.Dataset:
        """
        Compare the skill score between this trajectory and an `expected` trajectory.

        Parameters
        ----------

        expected : Dataset
            Another trajectory dataset, normally the observed / expected.

        method : str
            skill-score method, currently only liu-weissberg.

        **kwargs :
            Passed on to the skill-score method.

        Returns
        -------

        skill : DataArray
            The skill-score in the combined dimensions of both datasets.

        Notes
        -----
        Both datasets must have the same number of trajectories (N:N), or at least one of the datasets (normally the expected) must have a single trajectory only (1:N, N:1, 1:1).

        The datasets must be sampled (or have observations) at approximately the same timesteps. Consider using :meth:`gridtime` to interpolate one of the datasets to the other.

        Any additional dimensions will be broadcasted, so that the result include the combined dimensions of both datasets.

        Some skillscore methods (e.g. liu-weissberg) are not symmetrical. This specific skillscore is normalized on the length of the expected / observed trajectories. Thus `a.traj.skill(b)` will provide different numerical results than `b.traj.skill(a)`.

        Examples
        --------

        >>> import xarray as xr
        >>> import trajan as _
        >>> import lzma
        >>> b = lzma.open('examples/barents.nc.xz')
        >>> ds = xr.open_dataset(b)
        >>> expected = ds.copy()
        >>> ds = ds.traj.gridtime('1h')
        >>> expected = expected.traj.gridtime(ds.time)
        >>> skill = ds.traj.skill(expected)

        >>> skill  # Returns 1 since comparing to itself
        <xarray.DataArray 'Skillscore' (trajectory: 2)> Size: 16B
        array([1., 1.])
        Coordinates:
          * trajectory  (trajectory) int64 16B 0 1
        Attributes:
            method:   liu-weissberg


        >>> expected = ds.isel(trajectory=0)
        >>> skill = ds.traj.skill(expected)

        >>> skill
        <xarray.DataArray 'Skillscore' (trajectory: 2)> Size: 16B
        array([1.        , 0.60805799])
        Coordinates:
          * trajectory  (trajectory) int64 16B 0 1
        Attributes:
            method:   liu-weissberg
        """

    @abstractmethod
    def condense_obs(self) -> xr.Dataset:
        """
        Move all observations to the first index, so that the observation
        dimension is reduced to a minimum. When creating ragged arrays the
        observations from consecutive trajectories start at the observation
        index after the previous, causing a very long observation dimension.

        Original:

        .............. Observations --->
        trajectory 1: | t01 | t02 | t03 | t04 | t05 | nan | nan | nan | nan |
        trajectory 2: | nan | nan | nan | nan | nan | t11 | t12 | t13 | t14 |

        After condensing:

        .............. Observations --->
        trajectory 1: | t01 | t02 | t03 | t04 | t05 |
        trajectory 2: | t11 | t12 | t13 | t14 | nan |

        Returns:

            A new Dataset with observations condensed.
        """

    @abstractmethod
    def to_1d(self) -> xr.Dataset:
        """
        Convert dataset into a 1D dataset from. This is only possible if the
        dataset has a single trajectory.
        """

    @abstractmethod
    def to_2d(self, obs_dim='obs') -> xr.Dataset:
        """
        Convert dataset into a 2D dataset from.
        """

    @abstractmethod
    def append(self, da, obs_dims=None) -> xr.Dataset:
        """
        Append trajectories from other dataset to this.
        """

    def crop(self,
             lonmin=-360,
             lonmax=360,
             latmin=-90,
             latmax=90,
             shape='not_yet_implemented'):
        """
        Remove parts of trajectories outside of given geographical bounds.

        Parameters
        ----------

        lonmin : float
            Minimum longitude

        lonmax : float
            Maximum longitude

        latmin : float
            Minimum latitude

        latmax : float
            Maximum latitude

        Returns
        -------

        Dataset
            A new Xarray Dataset containing only given area
        """

        return self.ds.where((self.tlon > lonmin) & (self.tlon < lonmax)
                             & (self.tlat > latmin) & (self.tlat < latmax))

    def contained_in(self,
                     lonmin=-360,
                     lonmax=360,
                     latmin=-90,
                     latmax=90,
                     shape='not_yet_implemented'):
        """
        Return only trajectories fully within given geographical bounds.

        Parameters
        ----------

        lonmin : float
            Minimum longitude

        lonmax : float
            Maximum longitude

        latmin : float
            Minimum latitude

        latmax : float
            Maximum latitude

        Returns
        -------

        Dataset
            A new Xarray Dataset containing only the trajectories fully within given area.
        """

        condition = ((self.tlon.min(dim=self.obs_dim) > lonmin) &
                     (self.tlon.max(dim=self.obs_dim) < lonmax) &
                     (self.tlat.min(dim=self.obs_dim) > latmin) &
                     (self.tlat.max(dim=self.obs_dim) < latmax))

        return self.ds.isel({self.trajectory_dim: condition})

    def make_grid(self, dx, dy=None, z=None,
                  lonmin=None, lonmax=None, latmin=None, latmax=None):
        """
        Make a grid that covers all elements of dataset, with given spatial resolution.

        Parameters
        ----------

        dx : float
            Horizontal pixel/grid/cell size in meters, along direction of x or longitude

        dy : float
            Horizontal pixel/grid/cell size in meters, along direction of y or latitude
            If not provided, dy will be equal to dx (square pixels)

        z : array-like
            The bounds of the grid in the vertical.
            The outout grid will contain the centerpoints between each "layer", i.e. one element less

        lonmin, lonmax, latmin, latmax : float
            If not provided, these values will be taken from the min/max og lon/lat of the dataset

        Returns
        -------

        Dataset
            A new Xarray Dataset with the following coordinates:
              - time (only if input dataset has a time dimension)
              - lon, lat  (center of the cells)
              - lon_edges, lat_edges (edges of the cells, i.e. one element more than lon, lat)
              - z (vertical center of each layer, only if z if provided as input)
              - z_edges (vertical edges of each layer, i.e. one element more than z, and only if z if provided as input)
            The dataset contains the following variables on the above coordinates/grid:
              - layer_thickness: Thickness of layers in meters, i.e. the diff of input z
              - cell_area: Area [m2] of each grid cell
              - cell_volume: Volume [m3] of each grid cell, i.e. the above cell_area multiplied by layer_thickness
        """

        if dy is None:
            dy = dx  # Square pixels
        else:
            raise NotImplementedError('Rectangular pixels is not yet implemented')

        if lonmin is not None:
            if self.crs.is_geographic:
                xmin, xmax, ymin, ymax = lonmin, lonmax, latmin, latmax
            else:
                # Bounds are given as lon/lat, but dataset projection is not geographic
                # - Calculate (x,y) for all 4 corners (lon,lat)
                # - Use min/max og these (x,y) as bounds
                raise NotImplementedError('Not implemented')
        else:
            xmin = self.tx.min()
            xmax = self.tx.max()
            ymin = self.ty.min()
            ymax = self.ty.max()

        if self.crs.is_geographic:
            # dx is given in meters, but must be converted to degrees
            dy = dy / 111000
            dx = dy / np.cos(np.radians((ymin + ymax)/2)).values
            xdimname = 'lon'
            ydimname = 'lat'
        else:
            xdimname = 'x'
            ydimname = 'y'

        x = np.arange(xmin, xmax + dx*2, dx)  # One extra row/column
        y = np.arange(ymin, ymax + dy*2, dy)
        area = grid_area(x, y)

        # Create Xarray Dataset
        data_vars = {}
        data_vars['cell_area'] = ([ydimname, xdimname], area, {'long_name': 'Cell area', 'unit': 'm2'})

        coords = {ydimname: (y[0:-1]+y[1::])/2, xdimname: (x[0:-1]+x[1::])/2,
                  ydimname + '_edges': y, xdimname + '_edges': x}
        if 'time' in self.ds.coords:
            coords['time'] = self.ds.time
        if z is not None:
            z = np.array(z)
            coords['z'] = (z[0:-1] + z[1::]) / 2  # z refers to the center of layer
            coords['z_edges'] = z  # z_edges refer to the edges of each layer, i.a. one element longer
            data_vars['layer_thickness'] = (['z'], -(z[1::]-z[0:-1]), {'long_name': 'Layer thickness', 'unit': 'm'})

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        if z is not None:
            ds['cell_volume'] = ds.cell_area*ds.layer_thickness
            ds['cell_volume'].attrs = {'long_name': 'Cell volume', 'unit': 'm3'}

        return ds

    def concentration(self, grid, weights=None):
        """
        Calculate concentration of elements on a provided grid

        Parameters
        ----------

        grid : Dataset
            A grid Dataset, as returned from `trajan.make_grid`

        weights : string
            If provided, the concentration weighted with this variable will also be calculated

        Returns
        -------

        Dataset
            The same grid Xarray Dataset as input, with the following variables added:
              - number: the number of elements within each grid cell
              - number_area_concentration: the number of elements per area of each grid cell
                (only if grid does not contain z)
              - number_volume_concentration: the number of elements per volume of each grid cell
                (only if grid does not contain z)
              - <weights>_sum: the sum of property <weights> within each grid cell
              - <weights>_mean: the mean of property <weights> within each grid cell
              - <weights>_area_concentration: the concentration of <weights> per area of each grid cell
                (only if grid does not contain z)
              - <weights>_volume_concentration: the volume concentration of <weights> per volume of each grid cell
                (only if grid contains z)
        """

        # TODO: support other projections
        from xarray.groupers import BinGrouper, UniqueGrouper

        variables = ['lat', 'lon']  # Dataset variables needed for gridding
        if weights is not None:
            variables.append(weights)

        grid_dims = []
        if self.time_varname is not None:
            grid_dims.append('time')
        if 'z_edges' in grid.variables:
            variables.append('z')
            grid_dims.append('z')
        grid_dims = grid_dims + ['lat', 'lon']

        ds = self.ds[variables]  # Selecting subset for gridding

        groupers = {}
        if 'time' in grid_dims:
            groupers['time'] = UniqueGrouper()
        if 'z' in variables:
            groupers['z'] = BinGrouper(bins=np.flip(grid.z_edges))  # z must be increasing
        groupers['lat'] = BinGrouper(bins=grid.lat_edges)
        groupers['lon'] = BinGrouper(bins=grid.lon_edges)
        g = ds.groupby(groupers)

        # Calculate number concentration
        number = g.count()  # Number of elements per grid cell
        if 'z' in variables:
            number = number.isel(z_bins=slice(None, None, -1))  # Flipping z back
        grid['number'] = (grid_dims, number.lon.data)
        if 'z' in variables:
            grid['number_volume_concentration'] = grid.number / grid.cell_volume
        else:
            grid['number_area_concentration'] = grid.number / grid.cell_area

        # Calculate concentration of weights variable, if requested
        if weights is not None:
            w_sum = g.sum()
            if 'z' in variables:
                w_sum = w_sum.isel(z_bins=slice(None, None, -1))  # Flipping z back
            grid[weights + '_sum'] = (grid_dims, w_sum[weights].data)
            grid[weights + '_mean'] = grid[weights + '_sum'] / grid.number
            if 'z' in variables:  # Volume concentration
                grid[weights + '_volume_concentration'] = grid[weights + '_sum'] / grid.cell_volume
                grid[weights + '_volume_concentration'] = grid[weights + '_sum'] / grid.cell_volume
                grid[weights + '_volume_concentration'].attrs['long_name'] = f'{weights} per m3'
            else:  # Area concentration
                grid[weights + '_area_concentration'] = grid[weights + '_sum'] / grid.cell_area

        return grid
