import logging
from pathlib import Path
from trajan.readers.omb import read_omb_csv
import xarray as xr
import trajan as ta
import matplotlib.pyplot as plt


def test_read_csv_omb():
    path_to_test_data = Path.cwd() / "test_data" / "csv" / "omb1.csv"
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
