import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_barents(barents, plot):
    print(barents)

    barents.traj.animate()

    if plot:
        plt.show()

