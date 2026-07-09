import xarray as xr
import trajan as ta

def test_repr_orthogonal(openoil):
    repr = str(openoil.traj)
    assert '2015-11-16T00:00' in repr
    assert 'Timestep:        1:00:00' in repr
    assert "67 timesteps      [obs_dim: time]" in repr

def test_repr_ragged():
    ds = xr.open_dataset(ta.DATA_DIR + 'omb/bug32.nc')
    repr = str(ds.traj)
    assert '2023-10-19T15:46:53.514499520' in repr
