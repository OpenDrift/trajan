import trajan as ta
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_seltime(barents):
    print(barents)

    print(barents.time.min(skipna=True))
    print(barents.time.max(skipna=True))

    assert barents.time.min(skipna=True) < pd.to_datetime('2022-10-20')
    assert barents.time.max(
        skipna=True) > pd.to_datetime('2022-11-01T23:59:59')

    ds = barents.traj.seltime('2022-10-20', '2022-11-01')
    print(ds.time.min(skipna=True))
    print(ds.time.max(skipna=True))

    assert ds.time.min(skipna=True) >= pd.to_datetime('2022-10-20')
    assert ds.time.max(skipna=True) <= pd.to_datetime('2022-11-01T23:59:59')


def test_iseltime(barents):
    print(barents)
    # print(list(barents.groupby('trajectory')))

    ds = barents.traj.iseltime(0)
    assert ds.time.min(skipna=True) == barents.time.min(skipna=True)
    print(ds)

    assert ds.sizes['obs'] == 1
    assert np.all(~pd.isna(ds.time))

    print(ds)
    ds = barents.traj.iseltime(-1)
    assert ds.time.max(skipna=True) == barents.time.max(skipna=True)
    print(ds)
    assert ds.sizes['obs'] == 1
    assert np.all(~pd.isna(ds.time))

    ds = barents.traj.iseltime(slice(0, 2))
    print(ds)
    assert ds.sizes['obs'] == 2
    assert np.all(~pd.isna(ds.time))
