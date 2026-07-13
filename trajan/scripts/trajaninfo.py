# Utility script to quickly print summary information about a drifter collection file

import xarray as xr
import trajan as _
import click
from pathlib import Path
import lzma

@click.command()
@click.argument('tf')
@click.option('-p', is_flag=True, help="Plot the dataset")
@click.option('-l', is_flag=True, help="Show debug logging")


def main(tf, p, l):
    if l is True:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    tf = Path(tf)
    if tf.suffix == '.xz':
        with lzma.open(tf) as fd:
            ds = xr.open_dataset(fd)
            ds.load()
    else:
        ds = xr.open_dataset(tf)

    print(ds.traj)

    if p is True:
        ds.traj.plot(land='mask')
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == '__main__':
    main()
