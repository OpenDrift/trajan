"""
Reading and using a contiguous ragged dataset
================================================
"""

# %%

import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import trajan
import coloredlogs

# %%

# the test data: a small extract from the Sofar dataset https://sofar-spotter-archive.s3.amazonaws.com/index.html
path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "xr_spotter_bulk_test_data.nc"

xr_data = xr.load_dataset(path_to_test_data)

# this is a ragged contiguous dataset; the data variables are 1D arrays
xr_data

# %%

# trajan is able to plot etc
xr_data.traj.plot()
plt.show()

# %%

# it is possible to get a 2d-array dataset version using trajan:

# compare:
print(f"{xr_data.traj.ds = }")

# with:
xr_data_as_2darray = xr_data.traj.to_2d()
print(f"{xr_data_as_2darray = }")

# naturally, the 2darray version can be dumped to disk if you want:
xr_data_as_2darray.to_netcdf("./xr_spotter_bulk_test_data_as_2darray.nc")

# %%

