import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt
import pytest


def test_parcels_flatmesh(plot, parcels):
    ds = parcels
    ds = ds.traj.set_crs(None)
    print(ds)
    ds.traj.plot()

    if plot:
        plt.show()
    else:
        plt.close()

def test_parcels_flatmesh_animate(plot, parcels):
    ds = parcels
    ds = ds.traj.set_crs(None)
    print(ds)
    anim = ds.traj.animate()

    if plot:
        anim.show()
    else:
        fa = anim.build()
        fa._draw_was_started = True
        plt.close('all')
