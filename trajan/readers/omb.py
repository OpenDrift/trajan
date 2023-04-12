"""Utilities to import an OMB raw CSV file into Trajan"""

from pathlib import Path
import xarray as xr
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def read_omb_csv(path_in: Path) -> xr.Dataset:
    """todo"""

    # generic pandas read to be able to open from a variety of files
    
    # decode each entry;
    # only consider messages from the buoy to owner
    # only consider non empty messages
    # be verbose about messages that cannot be decoded: something is seriously wrong then!

    # turn into a trajan compatible format
