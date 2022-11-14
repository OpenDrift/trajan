# Utility script to quickly plot a drifter collection file

import sys
import xarray as xr
import trajan as ta

def main():
    tf = sys.argv[1]
    ds = xr.open_dataset(tf)
    ds.traj.plot()

if __name__ == '__main__':
    main()
