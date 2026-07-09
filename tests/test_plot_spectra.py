from trajan.readers.omb import read_omb_csv
import matplotlib.pyplot as plt
import trajan as ta


def test_plot_spectra_accessor(plot):
    csv_in = ta.DATA_DIR + 'omb/omb3.csv'
    ds = read_omb_csv(csv_in)
    print(ds)
    print(ds.elevation_energy_spectrum)
    print(ds.frequencies_waves_imu)

    plt.figure()
    ds.isel(trajectory=0).elevation_energy_spectrum.wave.plot(
        ds.isel(trajectory=0).time_waves_imu.squeeze())

    if plot:
        plt.show()

    plt.close('all')
