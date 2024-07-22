"""
Examples of leveraging xarray in combination with trajan 
================================================
"""

# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from trajan.readers.omb import read_omb_csv
import coloredlogs
import datetime
import matplotlib.pyplot as plt

# %%

# a few helper functions


def print_head(filename, nlines=3):
    """Print the first nlines of filename"""
    print("----------")
    print(f"head,{nlines} of {filename}")
    print("")
    with open(filename) as input_file:
        head = [next(input_file) for _ in range(nlines)]
    for line in head:
        print(line, end="")
    print("----------")


# %%

# adjust the level of information printed

coloredlogs.install(level='error')
# coloredlogs.install(level='debug')

# %%

# We start by preparing some example data to work with

# generate a trajan dataset
path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb3.csv"
xr_buoys = read_omb_csv(path_to_test_data)

# %%

# list all the buoys in the dataset

list_buoys = list(xr_buoys.trajectory.data)
print(f"{list_buoys = }")

# %%

# some users prefer to receive CSV files: dump all trajectories as CSV

# all positions to CSV, 1 CSV per buoy
for crrt_buoy in list_buoys:
    crrt_xr = xr_buoys.sel(trajectory=crrt_buoy)
    crrt_xr_gps = crrt_xr.swap_dims({'obs': 'time'})[["lat", "lon"]]
    crrt_xr_gps = crrt_xr_gps.dropna(dim='time')
    crrt_xr_gps.to_dataframe().to_csv(f"{crrt_buoy}_gps.csv")

print_head("drifter_1_gps.csv")

# %%

# similarly, some users prefer to receive CSV files for the wave information

# scalar data are easy to dump: the following creates files with
# all wave statistics to CSV, 1 file per buoy
for crrt_buoy in list_buoys:
    crrt_xr = xr_buoys.sel(trajectory=crrt_buoy)
    crrt_xr_wave_statistics = crrt_xr.swap_dims({"obs_waves_imu": "time_waves_imu"})[["pcutoff", "pHs0", "pT02", "pT24", "Hs0", "T02", "T24"]].rename({"time_waves_imu": "time"}).dropna(dim="time")
    crrt_xr_wave_statistics.to_dataframe().to_csv(f"{crrt_buoy}_wavestats.csv")

print_head("drifter_1_wavestats.csv")

# for spectra, we need to get the frequencies first and to label things
# all spectra to CSV
for crrt_buoy in list_buoys:
    crrt_xr = xr_buoys.sel(trajectory=crrt_buoy)
    # the how="all" is very important: since in the processed spectrum the "invalid / high noise" bins are set to NaN, we must only throw away the wave spectra for which all
    # bins are nan, but we must keep the spectra for which a few low frequency bins are nan.
    # if you want a CSV without any NaN, you can use "elevation_energy_spectrum" instead of "processed_elevation_energy_spectrum" to use the spectra with all bins, including
    # the bins that are dominated by low frequency noise
    crrt_xr_wave_spectra = crrt_xr.swap_dims({"obs_waves_imu": "time_waves_imu"})[["processed_elevation_energy_spectrum"]].rename({"time_waves_imu": "time"}).dropna(dim="time", how="all")
    
    list_frequencies = list(crrt_xr_wave_spectra.frequencies_waves_imu.data)
    
    for crrt_ind, crrt_freq in enumerate(list_frequencies):
        crrt_xr_wave_spectra[f"f={crrt_freq}"]=(['time'],  crrt_xr_wave_spectra["processed_elevation_energy_spectrum"][:, crrt_ind].data)
        
    crrt_xr_wave_spectra = crrt_xr_wave_spectra.drop_vars("processed_elevation_energy_spectrum").drop_dims("frequencies_waves_imu")
    crrt_xr_wave_spectra.to_dataframe().to_csv(f"{crrt_buoy}_wavespectra.csv", na_rep="NaN")

print_head("drifter_1_wavespectra.csv")

# %%

# it is easy to look into one single trajectory:

xr_specific_buoy = xr_buoys.sel(trajectory="drifter_1")

# make a restricted dataset about only the GPS data for a specific buoy, making GPS time the new dim
# this works because we now have a single buoy so there is only 1 "time" left
xr_specific_buoy_gps = xr_specific_buoy.swap_dims({'obs': 'time'})[["lat", "lon"]]
# keep only the valid GPS points: avoid the NaT that are due to time alignment in the initial nc file
xr_specific_buoy_gps = xr_specific_buoy_gps.dropna(dim='time')
xr_specific_buoy_gps.traj.plot()

plt.show()

# %%

