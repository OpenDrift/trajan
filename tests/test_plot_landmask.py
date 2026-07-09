import pytest
import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt


def test_defaults(openoil, plot):
    _, ax = openoil.traj.plot()

    if plot:
        plt.show()

    #return ax.figure


@pytest.mark.parametrize("land", ["auto", "c", "f", "mask"])
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=2.35)
def test_land_specs(openoil, plot, land):
    _, ax = openoil.traj.plot(land=land)

    if plot:
        plt.show()

    #return ax.figure
