import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_opendrift(openoil, plot):
    print(openoil)

    openoil.traj.plot()

    if plot:
        plt.show()
    else:
        plt.close()
