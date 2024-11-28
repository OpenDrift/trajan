"""
Plotting wave spectra data, using the accessor syntax
=============================================================================
"""

# %%

from pathlib import Path
from trajan.readers.omb import read_omb_csv
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
# by default, we "decorate", i.e. label axis etc

xr_data.isel(trajectory=0).processed_elevation_energy_spectrum.wave.plot(
    xr_data.isel(trajectory=0).time_waves_imu.squeeze(),
    )

plt.show()

# %%

# it is also possible to provide an axis on which to plot
# in this case, this will likely be part of a larger figure, and the user will likely want to put the
# labels etc themselves; remember to switch off decoration

# a plot with 3 lines, 2 columns
fig, ax = plt.subplots(3, 2)

ax_out, pclr = xr_data.isel(trajectory=0).processed_elevation_energy_spectrum.wave.plot(
    xr_data.isel(trajectory=0).time_waves_imu.squeeze(),
    # plot on the second line, first column
    ax=ax[1, 0],
    decorate=False,
    )

# the user can make the plot to their liking
ax[1, 0].set_xticks(ax[1, 0].get_xticks(), ax[1, 0].get_xticklabels(), rotation=45, ha='right')
ax[1,0].set_ylim([0.05, 0.25])
ax[1,0].set_ylabel("f [Hz]")

plt.tight_layout()

cbar = plt.colorbar(pclr, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('log$_{10}$(S) [m$^2$/Hz]')

plt.show()

# %%
