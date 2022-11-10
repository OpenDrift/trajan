import xarray as xr
import trajan as ta

# Extending xarray Dataset with functionality specific to trajectory datasets
# Presently supporting Cf convention H.4.1
# https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_multidimensional_array_representation_of_trajectories

@xr.register_dataset_accessor("traj")
class TrajAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, **kwargs):
        ta.plot(self._obj, **kwargs)
