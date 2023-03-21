import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_read_example_csv(drifter_csv, plot):
    ds = ta.read_csv(drifter_csv, name='Device')
    print(ds)

    ds.traj.plot()

    if plot:
        plt.show()
