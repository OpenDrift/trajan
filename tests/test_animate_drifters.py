import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_animate_barents(barents, plot):
    anim = barents.traj.animate()
    if plot:
        anim.show()
    else:
        fa = anim.build()
        fa._draw_was_started = True
        plt.close('all')

def test_animate_barents_color_by(barents, plot):
    speed = barents.traj.speed()
    anim = barents.traj.animate().color_by(speed, cmap='RdYlBu_r', vmin=0, vmax=2,
                                           label='Speed [m/s]')
    if plot:
        anim.show()
    else:
        fa = anim.build()
        fa._draw_was_started = True
        plt.close('all')

def test_animate_barents_show_trajectories(barents, plot):
    anim = barents.traj.animate().show_trajectories(alpha=0.2)
    if plot:
        anim.show()
    else:
        fa = anim.build()
        fa._draw_was_started = True
        plt.close('all')

def test_animate_barents_save_gif(barents, tmp_path):
    out = tmp_path / 'barents.gif'
    # land=None avoids cartopy shapefile download in CI
    barents.traj.animate(land=None).set_timestep('6h').save(str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close('all')

