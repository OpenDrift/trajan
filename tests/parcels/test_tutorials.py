import xarray as xr
import trajan as _
from datetime import timedelta
import matplotlib.pyplot as plt

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4

def test_moving_eddies(plot, tmpdir):
    fieldset = FieldSet.from_parcels("moving_eddies")

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=JITParticle, lon=[3.3e5,  3.3e5], lat=[1e5, 2.8e5])

    output_file = pset.ParticleFile(name=tmpdir / "EddyParticles.zarr", outputdt=timedelta(hours=1))
    pset.execute(AdvectionRK4, runtime=timedelta(days=6), dt=timedelta(minutes=5), output_file=output_file)

    ds = xr.open_dataset(tmpdir / 'EddyParticles.zarr', engine='zarr')
    ds.traj.plot()

    if plot:
        plt.show()
