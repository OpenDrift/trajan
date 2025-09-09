import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_animate_barents(barents, plot):

    anim = barents.traj.animate()

    if plot:
        plt.show()
    else:
        anim._draw_was_started = True  # To avoid warning
        plt.close('all')
