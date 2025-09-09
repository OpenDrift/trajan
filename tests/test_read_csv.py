import trajan as ta
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_read_example_csv(drifter_csv, plot):
    dc = pd.read_csv(drifter_csv)

    ds = ta.read_csv(drifter_csv, name='Device', time='Time', lon='Longitude', lat='Latitude')
    print(ds)

    assert len(dc) == ds.sizes['obs']

    assert ds.trajectory.attrs['cf_role'] == 'trajectory_id'
    assert ds.trajectory.attrs['long_name'] == 'trajectory name'

    ds.traj.plot()

    if plot:
        plt.show()
    plt.close('all')

def test_concat(drifter_csv):
    dc = pd.read_csv(drifter_csv)
    dc2 = dc.copy()
    dc2['Time'] = pd.to_datetime(dc['Time'], format='mixed') + pd.to_timedelta('1min')
    dc2['Device'] = 'drifter 2'

    dcc = pd.concat((dc, dc2))
    print(dcc)

    ds = ta.from_dataframe(dcc, name='Device', time='Time', lon='Longitude', lat='Latitude')
    print(ds)

    # obs dim is same as max of dc and dc2
    assert len(dc2) == ds.sizes['obs']

def test_lungard(test_data):
    b16 = pd.read_csv(test_data / 'csv/bug16.csv.xz')
    b23 = pd.read_csv(test_data / 'csv/bug23.csv.xz')

    dc = pd.concat((b16, b23))
    print(dc)

    ds = ta.from_dataframe(dc, name='Device', time='Time', lon='Longitude', lat='Latitude')

    print(ds)

    assert ds.sizes['obs'] == np.max((len(b16), len(b23)))

    assert ds.isel(trajectory=1).dropna('obs', how='all').sizes['obs'] == len(b16)
    assert ds.isel(trajectory=0).dropna('obs', how='all').sizes['obs'] == len(b23)

    dsg = ds.drop_vars(['Type', 'File'])
    dsg = dsg.traj.gridtime('1min')
    print(dsg)

    assert (np.isnan(ds.lon) == np.isnan(ds.lat)).all()
    assert (np.isnan(ds.isel(trajectory=0).lon) == np.isnan(ds.isel(trajectory=0).lat)).all()


def test_seals(test_data, tmpdir):
    s = test_data / 'csv/seals.csv.xz'
    ds = ta.read_csv(s, lat='Lat', lon='Lon', time='Timestamp', name='Instrument')
    print(ds)

    assert ds.sizes['trajectory'] == 5
    assert (ds['trajectory'].values == ['T1', 'T2', 'T3', 'T4', 'T5']).all()

    ds.to_netcdf(tmpdir / 'test.nc')

def test_condense_obs(test_data, tmpdir):
    s = test_data / 'csv/seals.csv.xz'
    ds = ta.read_csv(s, lat='Lat', lon='Lon', time='Timestamp', name='Instrument', __test_condense__=True)
    print(ds)

    assert ds.sizes['trajectory'] == 5
    assert (ds['trajectory'].values == ['T1', 'T2', 'T3', 'T4', 'T5']).all()

    assert ~np.isnan(ds['time'].isel(trajectory = 0).values[0])
    assert np.isnan(ds['time'].isel(trajectory = 1).values[0])

    ds2 = ds.traj.condense_obs()
    print(ds2)

    assert ds2.sizes['obs'] < ds.sizes['obs']

    for ti in range(ds.sizes['trajectory']):
        assert np.all(ds.isel(trajectory=ti).lon.dropna('obs') == ds2.isel(trajectory=ti).lon)
        assert np.all(ds.isel(trajectory=ti).lat.dropna('obs') == ds2.isel(trajectory=ti).lat)
        assert np.all(ds.isel(trajectory=ti).time.dropna('obs') == ds2.isel(trajectory=ti).time)

