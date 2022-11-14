# Utility script to quickly plot a drifter collection file

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta

def main():
    tf = sys.argv[1]
    ds = xr.open_dataset(tf)
    if 'status' in ds:  # hack for OpenDrift files
        ds = ds.where(ds.status>=0)
    ax, fig, gcrs = ta.plot(ds, show=False)
    start_time = np.nanmin(ds.time.data).astype('datetime64[s]')
    end_time = np.nanmax(ds.time.data).astype('datetime64[s]')
    name = tf
    ax.set_title(f'{name} [ {start_time} to {end_time} ]')
    plt.show()

if __name__ == '__main__':
    main()
