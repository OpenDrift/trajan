from .traj2d import Traj2d

class ContinuousRagged(Traj2d):
    def __init__(self, ds, obsdim, timedim, rowvar):
        ds_converted_to_traj2d = _convert_to_Traj2d(ds, obsdim, timedim, rowvar)
        super().__init__(ds_converted_to_traj2d, "obs", "time")


    def _convert_to_Traj2d(ds, obsdim, timedim, rowvar):
        # TODO: convert

        return ds_converted_to_traj2d
