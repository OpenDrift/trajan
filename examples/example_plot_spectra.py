"""
Plotting wave spectra data
================================================
"""

# %%

from pathlib import Path
from trajan.readers.omb import read_omb_csv
from trajan.plot.spectra import plot_trajan_spectra
import coloredlogs
import datetime

# adjust the level of information printed
coloredlogs.install(level='error')
# coloredlogs.install(level='debug')

# %%

# load the data from an example file with several buoys and a bit of wave spectra data
path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb3.csv"
xr_data = read_omb_csv(path_to_test_data)

# %%

# plot the spectra in one command
plot_trajan_spectra(xr_data)

# %%

# plot the spectra specifying more options by hand; it is also possible to automatically
# save the figure by setting fignamesave="some_name.png"
time_start = datetime.datetime(2022, 6, 16, 12, 30, 0)
time_end = datetime.datetime(2022, 6, 17, 8, 45, 0)
plot_trajan_spectra(xr_data, (time_start, time_end), (-2.0, 1.5))

# %%
