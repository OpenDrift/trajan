import xarray as xr
import logging

# recommended by cf-xarray
xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

from .traj import Traj
from .traj1d import Traj1d
from .traj2d import Traj2d
from .ragged import ContiguousRagged


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


@xr.register_dataset_accessor("traj")
class TrajA(Traj):
    def __new__(cls, ds):
        if 'traj' in ds.dims:
            logger.info(
                'Normalizing dimension name from "traj" to "trajectory".')
            ds = ds.rename({'traj': 'trajectory'})

        if 'trajectory' not in ds.dims:  # Add empty trajectory dimension, if single trajectory
            ds = ds.expand_dims({'trajectory': 1})
            ds['trajectory'].attrs['cf_role'] = 'trajectory_id'

        obsdim = None
        timedim = None

        tx = detect_tx_dim(ds)

        # if we have a 1D dims, this is most likely some contiguous data
        # there may be a few exceptions though, so be ready to default to the classical 2D parser below
        if len(tx.dims) == 1:
            # only support ContiguousRagged for now
            ocls = ContiguousRagged
            
            # we have a dataset where data are stored in 1D array
            # NOTE: this is probably not standard; something to point to the CF conventions?
            if "index" in tx.dims:
                obsdim = "index"

                # find all variables with standard_name "time"
                with_standard_name_time = ds.cf[["time"]]
                # find the list of these that have the ("index",) dimension
                with_standard_name_time_and_dim_index = \
                    [with_standard_name_time[var].name for var in with_standard_name_time.coords \
                                                       if with_standard_name_time[var].dims == ("index",)]
                # if there is more than one such, this is ambiguoys
                if len(with_standard_name_time_and_dim_index) == 1:
                    timecoord = with_standard_name_time_and_dim_index[0]
                else:
                    raise ValueError(f"cannot deduce the timecoord; we have the following candidates for timecoord: {with_standard_name_time_and_dim_index = }")

                trajectorycoord = ds.cf["trajectory_id"].name

                # NOTE: this is probably not standard; something to point to the CF conventions?
                if "rowsize" in ds.data_vars:
                    rowsizevar = "rowsize"
                else:
                    raise ValueError("cannot find rowsizevar in 1D-array dataset")

                logger.debug(
                    f"1D storage dataset; detected: {obsdim = }, {timecoord = }, {trajectorycoord = }, {rowsizevar}"
                )

                return ocls(ds, obsdim, timecoord, trajectorycoord, rowsizevar)

            else:
                logging.warning(f"{ds} has {tx.dims = } which is of dimension 1 but is not index; this is a bit unusual; try to parse with Traj1d or Traj2d")

        # we have a ds where 2D arrays are used to store data, this is either Traj1d or Traj2d
        # there may also be some slightly unusual cases where these Traj1d and Traj2d classes will be used on data with 1D arrays
        if 'obs' in tx.dims:
            obsdim = 'obs'
            timedim = detect_time_dim(ds, obsdim)

        elif 'time' in tx.dims:
            obsdim = 'time'
            timedim = 'time'

        else:
            for d in tx.dims:
                if not ds[d].attrs.get(
                        'cf_role',
                        None) == 'trajectory_id' and not 'traj' in d:

                    obsdim = d
                    timedim = detect_time_dim(ds, obsdim)

                    break

            if obsdim is None:
                logger.warning('No time or obs dimension detected.')

        logger.debug(
            f"2D storage dataset; detected obs-dim: {obsdim}, detected time-dim: {timedim}.")

        if obsdim is None:
            ocls = Traj1d

        elif len(ds[timedim].shape) <= 1:
            logger.debug('Detected structured (1D) trajectory dataset')
            ocls = Traj1d

        elif len(ds[timedim].shape) == 2:
            logger.debug('Detected un-structured (2D) trajectory dataset')
            ocls = Traj2d

        else:
            raise ValueError(
                f'Time dimension has shape greater than 2: {ds["timedim"].shape}'
            )

        return ocls(ds, obsdim, timedim)
