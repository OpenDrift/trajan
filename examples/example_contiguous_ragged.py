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

# this is a contiguous ragged dataset; the data variables are 1D arrays
xr_data

# %%

# trajan is able to plot etc
xr_data.traj.plot()
plt.show()

# %%

# Convert ContiguousRaggedArray to RaggedArray (non-contiguous) dataset:

# compare:
print(f"{xr_data = }")

# with:
xr_data_as_ragged = xr_data.traj.to_ragged()
print(f"{xr_data_as_ragged = }")

# Dumping the Ragged dataset to disk:
xr_data_as_ragged.to_netcdf("./xr_spotter_bulk_test_data_as_ragged.nc")

# %%

