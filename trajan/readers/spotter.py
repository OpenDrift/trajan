from pathlib import Path
import pandas as pd
import xarray as xr
import trajan as ta
import logging

logger = logging.getLogger(__name__)

def read_spotter_csv(path: Path):
    logger.debug(f'reading spotter file from: {path}..')

    df = pd.read_csv(path)
    ds = ta.from_dataframe(df, name='trajectory')
    return ds
