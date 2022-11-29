# Utility script to quickly plot a drifter collection file

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta
import click
from pathlib import Path
import lzma

@click.command()
@click.argument('tf')
@click.option('-l', '--land',
              default='auto',
              help='Scale of coastline (f, h, i, l, c or mask)',
              type=str,
              multiple=False)
@click.option('-m', '--margin',
              default=.1,
              help='Margin/whitespace in degrees around drifter coverage',
              type=float,
              multiple=False)
@click.option('-s', '--start_time',
              default=None,
              help='Show only data after this time',
              type=str,
              multiple=False)
@click.option('-e', '--end_time',
              default=None,
              help='Show only data before this time',
              type=str,
              multiple=False)

def main(tf, land, start_time, end_time, margin):
    tf = Path(tf)
    if tf.suffix == '.xz':
        with lzma.open(tf) as fd:
            ds = xr.open_dataset(fd)
            ds.load()
    else:
        ds = xr.open_dataset(tf)

    if 'status' in ds:  # hack for OpenDrift files
        ds = ds.where(ds.status>=0)

    trajectory_names = None
    color = 'gray'
    dsub = ds.filter_by_attrs(cf_role='trajectory_id')
    if len(dsub.data_vars) == 1:
        trajectory_names = dsub.to_array().values[0]
        color = None

    if start_time is not None:
        ds = ds.where(ds.time>np.datetime64(start_time))
    if end_time is not None:
        ds = ds.where(ds.time<np.datetime64(end_time))

    ds.traj.plot(label=trajectory_names, color=color, land=land, margin=margin)

    start_time = np.nanmin(ds.time.data).astype('datetime64[s]')
    end_time = np.nanmax(ds.time.data).astype('datetime64[s]')
    name = tf

    plt.gca().set_title(f'{name} [ {start_time} to {end_time} ]')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
