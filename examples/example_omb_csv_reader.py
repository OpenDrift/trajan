import logging
from pathlib import Path
from trajan.readers.omb import read_omb_csv
import xarray as xr

if __name__ == "__main__":
    import trajan as ta
    import matplotlib.pyplot as plt

    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.ERROR)

    path_to_test_data = Path.cwd().parent / "tests" / "test_data" / "csv" / "omb1.csv"
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

    print(xr_result)
    xr_result.to_netcdf("test.nc")

    ds = xr.open_dataset('test.nc')
    ds.traj.plot(margin=1, land='mask', color=None, label=ds.drifter_names.values)
    plt.show()
