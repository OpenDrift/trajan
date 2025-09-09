import pytest
import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt


def test_defaults(opendrift_sim, plot):
    opendrift_sim = opendrift_sim.where(
        opendrift_sim.status >= 0)  # only active particles
    opendrift_sim.traj.plot()

    if plot:
        plt.show()
    else:
        plt.close()


@pytest.mark.parametrize("land", ["auto", "c", "f", "mask"])
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=2.35)
def test_land_specs(opendrift_sim, plot, land):
    opendrift_sim = opendrift_sim.where(
        opendrift_sim.status >= 0)  # only active particles
    opendrift_sim.traj.plot(land=land)

    if plot:
        plt.show()
    else:
        plt.close()
