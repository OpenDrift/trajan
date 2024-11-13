import xarray as xr
import trajan as _

def test_repr_1d(opendrift_sim):
    repr = str(opendrift_sim.traj)
    assert '2015-11-16T00:00' in repr
    assert 'Timestep:       1:00:00' in repr
    assert "67 timesteps    time['time'] (1D)" in repr

def test_repr_2d(test_data):
    ds = xr.open_dataset(test_data / 'bug32.nc')
    repr = str(ds.traj)
    assert '2023-10-19T15:46:53.514499520' in repr
