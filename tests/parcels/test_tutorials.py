import xarray as xr
import trajan as _
from datetime import timedelta
import matplotlib.pyplot as plt

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from . import moving_eddies_fieldset

def test_moving_eddies_xy(plot, tmpdir, moving_eddies_fieldset):
    pset = ParticleSet.from_list(fieldset=moving_eddies_fieldset, pclass=JITParticle, lon=[3.3e5,  3.3e5], lat=[1e5, 2.8e5])

    output_file = pset.ParticleFile(name=tmpdir / "EddyParticles.zarr", outputdt=timedelta(hours=1))
    pset.execute(AdvectionRK4, runtime=timedelta(days=6), dt=timedelta(minutes=5), output_file=output_file)

    ds = xr.open_dataset(tmpdir / 'EddyParticles.zarr', engine='zarr')
    ds = ds.traj.set_crs(None)
    print(ds)
    ds.traj.plot()

    if plot:
        plt.show()
