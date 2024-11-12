# Utility script to quickly print summary information about a drifter collection file

import xarray as xr
import trajan as _
import click
from pathlib import Path
import lzma

@click.command()
@click.argument('tf')

def main(tf):
    tf = Path(tf)
    if tf.suffix == '.xz':
        with lzma.open(tf) as fd:
            ds = xr.open_dataset(fd)
            ds.load()
    else:
        ds = xr.open_dataset(tf)

    if 'status' in ds:  # hack for OpenDrift files
        ds = ds.where(ds.status>=0)

    print(ds.traj)

if __name__ == '__main__':
    main()
