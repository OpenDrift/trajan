"""
Plotting wave spectra data, using the accessor syntax
================================================
"""

# %%

exit()

# %%

ipython3

# %%

from pathlib import Path
from trajan.readers.omb import read_omb_csv
from trajan.plot.spectra import plot_trajan_spectra
import coloredlogs
import datetime
import matplotlib.pyplot as plt

# adjust the level of information printed
# coloredlogs.install(level='error')
coloredlogs.install(level='debug')

# %%

# load the data from an example file with several buoys and a bit of wave spectra data
path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb3.csv"
xr_data = read_omb_csv(path_to_test_data)

# %%

# if no axis is provided, an axis will be generated automatically

xr_data.isel(trajectory=0).processed_elevation_energy_spectrum.wave.plot(
    xr_data.isel(trajectory=0).time_waves_imu.squeeze(),
    )

plt.show()

# %%

# it is also possible to provide an axis on which to plot

# a plot with 3 lines, 2 columns
fig, ax = plt.subplots(3, 2)

ax_out = xr_data.isel(trajectory=0).processed_elevation_energy_spectrum.wave.plot(
    xr_data.isel(trajectory=0).time_waves_imu.squeeze(),
    # plot on the second line, first column
    ax=ax[1, 0]
    )

plt.show()

# %%
