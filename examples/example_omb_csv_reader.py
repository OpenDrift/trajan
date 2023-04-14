"""
Reading an OMB Rock7 CSV file into trajan
================================================
"""

from pathlib import Path
from trajan.readers.omb import read_omb_csv
import trajan as ta
import coloredlogs

# adjust the level of information printed
# coloredlogs.install(level='error')
coloredlogs.install(level='debug')

#%%
# example 1: default configuration of the wave packets

path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb1.csv"

# for most users, using the default spectra packet format and this will be enough
xr_result = read_omb_csv(path_to_test_data)

# remember to add the CF required metadata for your specific deployment
xr_result = xr_result.assign_attrs(
    {
        "creator_name": "your name",
        "creator_email": "your email",
        "title": "a descriptive title",
        "summary": "a descriptive summary",
        "anything_else": "corresponding data",
    }
)

# look at the dataset obtained
print(xr_result)

#%%
# example 2: custom size of wave packets; for users who have changed the firmware to transmit
# more or less spectrum bins

path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb2.csv"

# the properties description of the modified binary wave packets
dict_wave_packet_params = {
    "_BD_YWAVE_PACKET_MIN_BIN": 9,
    "_BD_YWAVE_PACKET_MAX_BIN": 94,
    "LENGTH_FROM_SERIAL_OUTPUT": 198,
}

# specify the binary wave packet specification corresponding to the modified firmware
xr_result = read_omb_csv(path_to_test_data, modified_wave_packet_properties=dict_wave_packet_params)

# look at the dataset obtained
print(xr_result)

# from there on, all is similar to above
