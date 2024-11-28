"""
Examples of compressing data when dumping to .nc
==============================================================================
"""

# %%

import xarray as xr
from trajan.readers.omb import read_omb_csv
from pathlib import Path

# %%

path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb_large.csv"
xr_buoys = read_omb_csv(path_to_test_data)

# %%

# by default, to_netcdf does not perform any compression
xr_buoys.to_netcdf("no_compression.nc")

# on my machine, this is around 33MB
!ls -lh no_compression.nc

# %%

# one can perform compression by providing explicitly the right arguments
# note that the best way to compress may depend on your dataset, the access
# pattern you want to be fastest, etc - be aware of memory layout and
# performance!

# a simple compression, on a per-trajectory basis: each trajectory will
# be compressed as a chunk, this means that it will be fast to retrieve one
# full trajectory, but slow to retrieve e.g. the 5th point of all trajectories.

# choose the encoding chunking - this may be application dependent, here
# chunk trajectory as a whole
def generate_chunksize(var):
    dims = xr_buoys[var].dims
    shape = list(xr_buoys[var].shape)

    idx_trajectory = dims.index("trajectory")
    shape[idx_trajectory] = 1

    return tuple(shape)
    

# set the encoding for each variable
encoding = {
    var: {"zlib": True, "complevel": 5, "chunksizes": generate_chunksize(var)} \
        for var in xr_buoys.data_vars
}

# the encoding looks like:
for var in encoding:
    print(f"{var}: {encoding[var] = }")

# dump, this time with compression
xr_buoys.to_netcdf("trajectory_compression.nc", encoding=encoding)

# on my machine, this is around 5.6MB
!ls -lh trajectory_compression.nc

# %%
