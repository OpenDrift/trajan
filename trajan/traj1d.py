from .traj import Traj

class Traj1d(Traj):
    """
    A structured dataset, where each trajectory is always given at the same times. Typically the output from a model or from a gridded dataset.
    """

    def __init__(self, ds):
        super().__init__(ds)


    def time_to_next(self):
        time_step = self.ds.time[1] - self.ds.time[0]
        return time_step
