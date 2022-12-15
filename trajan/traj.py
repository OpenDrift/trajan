import xarray as xr

class Traj:
    ds: xr.Dataset

    def __init__(self, ds):
        self.ds = ds
