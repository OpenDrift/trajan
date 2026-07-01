# Trajectory analysis ( TrajAn )

[![Build (python)](https://github.com/OpenDrift/trajan/workflows/Python/badge.svg)](https://github.com/OpenDrift/trajan/actions?query=branch%3Amain)
[![Docs](https://github.com/OpenDrift/trajan/workflows/Docs/badge.svg)](https://github.com/OpenDrift/trajan/actions?query=branch%3Amain)
[![PyPI version](https://badge.fury.io/py/trajan.svg)](https://badge.fury.io/py/trajan)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/trajan/badges/version.svg)](https://anaconda.org/conda-forge/trajan)

TrajAn is a Python package for working with trajectory datasets that follow the [CF-conventions for trajectories](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#trajectory-data). Trajectory datasets contain position time series from e.g. drifting buoys or output from Lagrangian models.

![Barents Sea drifter trajectories coloured by speed](https://opendrift.github.io/trajan/_images/sphx_glr_example_drifters_004.png)

## Installation

**conda / mamba** (recommended):
```bash
mamba install -c conda-forge trajan
```

**pip**:
```bash
pip install trajan
```

## Quick start

TrajAn exposes a `.traj` accessor on xarray Datasets:

```python
import lzma
import xarray as xr
import trajan as ta

# Open a CF-trajectory dataset (e.g. from drifting buoys)
with lzma.open("barents.nc.xz") as f:
    ds = xr.open_dataset(f)
    ds.load()

# Basic map plot — geographic projection is chosen automatically
ds.traj.plot()

# Calculate drifter speed, drop unreliable fixes, and plot coloured by speed
ds = ds.traj.drop_where(ds.traj.time_to_next() < np.timedelta64(5, "m"))
speed = ds.traj.speed()
ds.traj.plot(color=speed)

# Interpolate to a regular 1-hour time grid
dh = ds.traj.gridtime("1h")

# Animate the trajectories
ds.traj.animate().show()
```

For more examples and the full API reference see the **[documentation](https://opendrift.github.io/trajan/gallery)**.

