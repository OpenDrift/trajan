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

# this is because, under the hood, trajan is generating a Traj2D non ragged dataset that follows the usual trajan conventions

xr_data.traj.ds

# %%

