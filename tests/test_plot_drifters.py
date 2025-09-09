import trajan as ta
import xarray as xr
import matplotlib.pyplot as plt

def test_barents(barents, plot):
    print(barents)

    barents.traj.plot()

    if plot:
        plt.show()
    else:
        plt.close()

def test_barents_linecolor(barents, plot):
    speed = barents.traj.speed()
    mappable = barents.traj.plot(color=speed.where(speed<10),
                                 linewidth=2, land='h', margin=.4)

    cb = plt.gcf().colorbar(mappable,
              orientation='horizontal',
              pad=.05,
              aspect=30,
              shrink=.8,
              drawedges=False)
    cb.set_label('Drifter speed  [m/s]')

    if plot:
        plt.show()
    else:
        plt.close()
