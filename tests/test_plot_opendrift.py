import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_opendrift(opendrift_sim, plot):
    print(opendrift_sim)

    opendrift_sim = opendrift_sim.where(opendrift_sim.status>=0)  # only active particles
    opendrift_sim.traj.plot()

    if plot:
        plt.show()
