# Create a sample trajectory dataset for demonstrating and testing of Trajan package
# This requires that OpenDrift is installed

from datetime import datetime, timedelta
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.openoil import OpenOil

o = OpenOil(loglevel=20)

# Add forcing
reader_arome = reader_netCDF_CF_generic.Reader(o.test_data_folder() +
    '16Nov2015_NorKyst_z_surface/arome_subset_16Nov2015.nc')
reader_norkyst = reader_netCDF_CF_generic.Reader(o.test_data_folder() +
    '16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc')
o.add_reader([reader_norkyst, reader_arome])

# Seeding some particles, continuous spill
o.seed_elements(lon=4.2, lat=60.1, radius=1000, number=1000,
                time=[reader_arome.start_time, reader_arome.end_time])

# Running model
o.run(end_time=reader_norkyst.end_time,
      export_variables=['viscosity', 'z'], outfile='openoil.nc')

o.plot(fast=True, linecolor='viscosity')
