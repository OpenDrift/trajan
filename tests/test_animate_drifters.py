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
    anim = barents.traj.animate().color_by(speed, cmap='plasma', vmin=0, vmax=2,
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

def test_animate_barents_save_mp4(barents, tmp_path):
    import matplotlib.animation
    out = tmp_path / 'barents.mp4'
    if not matplotlib.animation.FFMpegWriter.isAvailable():
        import pytest
        pytest.skip('ffmpeg not available')
    barents = barents.traj.iseltime(slice(0, 10))
    speed = barents.traj.speed()
    # land=None avoids cartopy shapefile download in CI
    (barents.traj.animate(land=None)
        .color_by(speed, cmap='plasma', vmin=0, vmax=1, label='Speed [m/s]')
        .set_timestep('6h')
        .save(str(out)))
    assert out.exists() and out.stat().st_size > 0
    plt.close('all')

