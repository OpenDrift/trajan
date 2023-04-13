import logging
from pathlib import Path
from trajan.readers.omb import read_omb_csv
# import xarray as xr
# import trajan as ta
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.ERROR)

    ####################
    # example 1: default
    if True:
        path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb1.csv"

        # for most users, using the default spectra packet format and this will be enough
        xr_result = read_omb_csv(path_to_test_data)
        xr_result = xr_result.assign_attrs(
            {
                "creator_name": "your name",
                "creator_email": "your email",
                "title": "a descriptive title",
                "summary": "a descriptive summary",
                "anything_else": "corresponding data",
            }
        )

        # print(xr_result)
        # xr_result.to_netcdf("test.nc")

        # ds = xr.open_dataset('test.nc')
        # ds.traj.plot(margin=1, land='mask', color=None, label=ds.drifter_names.values)
        # plt.show()

    ####################
    # example 2: custom size of wave packets; for users who have changed the firmware to transmit
    # more or less spectrum bins
    if True:
        path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb2.csv"

        dict_wave_packet_params = {
            "_BD_YWAVE_PACKET_MIN_BIN": 9,
            "_BD_YWAVE_PACKET_MAX_BIN": 94,
            "LENGTH_FROM_SERIAL_OUTPUT": 198,
        }

        # if necessary, the user can specify the binary wave packet specification, if this is a modified firmware
        xr_result = read_omb_csv(path_to_test_data, modified_wave_packet_properties=dict_wave_packet_params)

        # print(xr_result)
