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

    ds = barents.traj.seltime('2022-10-20', '2022-11-01')
    print(ds.time.min(skipna=True))
    print(ds.time.max(skipna=True))

    assert ds.time.min(skipna=True) >= pd.to_datetime('2022-10-20')
    assert ds.time.max(skipna=True) <= pd.to_datetime('2022-11-01')


def test_iseltime(barents):
    print(barents)
    # print(list(barents.groupby('trajectory')))

    ds = barents.traj.iseltime(0).expand_dims('obs')
    assert ds.time.min(skipna=True) == barents.time.min(skipna=True)

    assert ds.dims['obs'] == 1
    assert np.all(~pd.isna(ds.time))

    print(ds)
    ds = barents.traj.iseltime(-1).expand_dims('obs')
    assert ds.time.max(skipna=True) == barents.time.max(skipna=True)
    print(ds)
    assert ds.dims['obs'] == 1
    assert np.all(~pd.isna(ds.time))

    ds = barents.traj.iseltime(slice(0,2))
    print(ds)
    assert ds.dims['obs'] == 2
    assert np.all(~pd.isna(ds.time))
