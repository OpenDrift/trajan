import xarray as xr
import logging
import numpy as np

# recommended by cf-xarray
xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

from .traj import Traj, detect_tx_variable
from .traj1d import Traj1d
from .traj2d import Traj2d
from .ragged import ContiguousRagged


def detect_time_variable(ds, obs_dim):
    logger.debug(f'Detecting time-variable for "{obs_dim}"..')
    # TODO: should use cf-xarray here
    for v in ds.variables:
        if obs_dim in ds[v].dims and 'time' in v:
            return v

    raise ValueError("No time variable detected")

def detect_trajectory_dim(ds):
    logger.debug('Detecting trajectory dimension')
    if 'trajectory_id' in ds.cf.cf_roles:  # This is the proper CF way
        trajectory_var = ds.cf['trajectory_id']
        if trajectory_var.name in trajectory_var.sizes:  # Check that this is a dimension
            return trajectory_var.name
        else:
            if len(trajectory_var.dims) > 1:
                logger.debug(f'trajectory_id is {trajectory_var.name}, but dimensions '
                               'have other names: {str(list(trajectory_var.sizes))}')
            elif len(trajectory_var.dims) == 1:  # Using the single dimension name
                return list(trajectory_var.sizes)[0]
            else:
                logger.debug('Single trajectory, a trajectory dimension will be added')
                return None

    logger.debug('No trajectory_id attribute/variable found, trying to identify by name.')
    tx = detect_tx_variable(ds)
    for tdn in ['trajectory', 'traj']:  # Common names of trajectory dimension
        if tdn in tx.dims:
            return tdn

    return None  # Did not succeed in detecting trajectory dimension


@xr.register_dataset_accessor("traj")
class TrajA(Traj):
    def __new__(cls, ds):

        trajectory_dim = detect_trajectory_dim(ds)

        if trajectory_dim is None:
            if 'trajectory_id' in ds.cf.cf_roles:
                trajectory_id = ds.cf.cf_roles['trajectory_id']
                if len(trajectory_id) > 1:
                        raise ValueError(f'Dataset has several trajectory_id variables: {trajectory_id}')
                else:
                    trajectory_id = trajectory_id[0]
                    logger.debug(f'Using trajectory_id variable name ({trajectory_id}) '
                                   'as trajectory dimension name')
                    trajectory_dim = trajectory_id
                    ds = ds.set_coords(trajectory_dim)
                    ds = ds.expand_dims(trajectory_dim, create_index_for_new_dim=False)
            else:
                logger.debug('Creating new trajectory dimension "trajectory"')
                trajectory_dim = 'trajectory'
                ds = ds.expand_dims({trajectory_dim: 1})
                ds[trajectory_dim].attrs['cf_role'] = 'trajectory_id'

        obs_dim = None
        time_varname = None

        tx = detect_tx_variable(ds)

        # if we have a 1D dims, this is most likely some contiguous data
        # there may be a few exceptions though, so be ready to default to the classical 2D parser below
        if len(tx.dims) == 1:
            # only support ContiguousRagged for now
            ocls = ContiguousRagged

            # we have a dataset where data are stored in 1D array
            # NOTE: this is probably not standard; something to point to the CF conventions?
            # NOTE: for now, there is no discovery of the "index" dim, this is hardcorded; any way to do better?
            if "index" in tx.dims:
                obs_dim = "index"

                # discover the timecoord variable name #######################
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
                    raise ValueError(f"cannot deduce the timecoord; we have the following candidates: {with_standard_name_time_and_dim_index = }")

                # KFD TODO: the below detection should be generalized to dynamic dimension names

                # discover the "rowsize" variable name #######################
                # NOTE: this is probably not standard; something to point to the CF conventions? should we need a standard_name for this, instead of the following heuristics?
                # find all variables with the ("trajectory", ) dimension
                with_dim_trajectory = \
                    [ds.data_vars[var].name for var in ds.data_vars if ds[var].dims == ("trajectory",)]
                if len(with_dim_trajectory) == 1:
                    rowsizevar = with_dim_trajectory[0]
                else:
                    raise ValueError(f"cannot deduce rowsizevar; we have the following candidates: {with_dim_trajectory = }")
                # sanity check
                if not np.sum(ds[rowsizevar].to_numpy()) == len(ds[obs_dim]):
                    raise ValueError("mismatch between the index length and the sum of the deduced trajectory lengths")

                logger.debug(
                    f"1D storage dataset; detected: {obs_dim = }, {timecoord = }, {trajectory_dim = }, {rowsizevar}"
                )

                return ocls(ds, trajectory_dim, obs_dim, timecoord, rowsizevar)

            else:
                logging.debug(f"{ds} has {tx.dims = } which is of dimension 1 but is not index; this is a bit unusual; try to parse with Traj1d or Traj2d")

        # we have a ds where 2D arrays are used to store data, this is either Traj1d or Traj2d
        # there may also be some slightly unusual cases where these Traj1d and Traj2d classes will be used on data with 1D arrays
        if 'obs' in tx.dims:
            obs_dim = 'obs'
            time_varname = detect_time_variable(ds, obs_dim)

        elif 'index' in tx.dims:
            obs_dim = 'obs'
            time_varname = detect_time_variable(ds, obs_dim)

        elif 'time' in tx.dims:
            obs_dim = 'time'
            time_varname = 'time'

        else:
            for d in tx.dims:
                if not ds[d].attrs.get(
                        'cf_role',
                        None) == 'trajectory_id' and not 'traj' in d:

                    obs_dim = d
                    time_varname = detect_time_variable(ds, obs_dim)

                    break

            if obs_dim is None:
                logger.debug('No time or obs dimension detected.')

        logger.debug(
            f"Detected obs-dim: {obs_dim}, detected time-variable: {time_varname}.")

        if obs_dim is None:
            ocls = Traj1d

        elif len(ds[time_varname].shape) <= 1:
            logger.debug('Detected structured (1D) trajectory dataset')
            ocls = Traj1d

        elif len(ds[time_varname].shape) == 2:
            logger.debug('Detected un-structured (2D) trajectory dataset')
            ocls = Traj2d

        else:
            raise ValueError(
                    f'Time variable has more than two dimensions: {time_varname}: {ds[time_varname].shape} / {ds[time_varname].dims}'
            )

        # TODO: The provided attributes could perhaps be added here before returning
        return ocls(ds, trajectory_dim, obs_dim, time_varname)
