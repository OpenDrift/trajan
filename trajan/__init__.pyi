"""
Stub file for trajan public API.

Declares ``trajan.Dataset`` — a typed alias for ``xr.Dataset`` that exposes
the ``.traj`` accessor.  Use it as a type annotation to get LSP completion:

    import trajan
    ds: trajan.Dataset = xr.open_dataset("file.nc")
    ds.traj.speed()   # <- full completion
"""

import pandas as pd
import xarray as xr
from typing import Any

from .traj import Traj as Traj
from .traj1d import Traj1d as Traj1d
from .traj2d import Traj2d as Traj2d

class Dataset(xr.Dataset):
    """xarray Dataset with the trajan ``.traj`` accessor typed."""

    @property
    def traj(self) -> Traj: ...

def versions() -> str: ...

def read_csv(f: Any, **kwargs: Any) -> Dataset: ...

def from_dataframe(
    df: pd.DataFrame,
    lon: str = ...,
    lat: str = ...,
    time: str = ...,
    name: str | None = ...,
    *,
    __test_condense__: bool = ...,
) -> Dataset: ...

def trajectory_dict_to_dataset(
    trajectory_dict: dict[str, Any],
    variable_attributes: dict[str, Any] | None = ...,
    global_attributes: dict[str, Any] | None = ...,
) -> Dataset: ...
