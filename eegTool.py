import argparse
import os

import matplotlib as plt
import matplotlib.pyplot as plt
import mne
import numpy as np
import pyprep
import yasa
from fooof import FOOOFGroup
from fooof.bands import Bands
from matplotlib import cm
from mne.time_frequency import psd_welch


class ParserPreparation:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Program for visualization EEG data in different band power. "
                                                     "It save bandPower and convert mff to fif.",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        required = parser.add_argument_group('Required arguments')
        optional = parser.add_argument_group('Optional arguments')

        required.add_argument("-i", "--input", help="input file path", required=True, type=str)
        required.add_argument("-o", "--out-dir-path", help="output directory path. If results exist will be "
                                                           "overwritten.",
                              required=True, type=str)
        optional.add_argument("-b", "--begin-time", help="Start time of the raw data to use in seconds (must be >= 0).",
                              required=False, type=int, default=25, const=25, nargs='?')
        optional.add_argument("-e", "--end-time", help="End time of the raw data to use in seconds (cannot exceed "
                                                       "data duration).",
                              required=False, type=int)

        optional.add_argument("-s", "--sfreq", help="New sample rate to use.",
                              required=False, type=int, default=None, const=256, nargs='?')

        optional.add_argument("-fmin", "--frequency-min", help="Minimum value of filtering frequency",
                              required=False, type=int, default=1, const=1, nargs='?')
        optional.add_argument("-fmax", "--frequency-max", help="Max value of filtering frequency",
                              required=False, type=int, default=45, const=45, nargs='?')

        optional.add_argument("-v", "--verbose", help="Control verbosity of the logging output. If None, use the "
                                                      "default verbosity level.", type=bool, default=False,
                              nargs='?', required=False, const=True)
        optional.add_argument('-ce', '--chart-extension', help="Chart extension",
                              type=str, default="eps", const="eps", nargs='?')
        optional.add_argument('-c', '--convert', help="If true, will convert to fif file.", type=bool, default=True,
                              nargs='?', required=False, const=True)

        self.args = parser.parse_args()

    def getArgs(self):
        return self.args


def plot_band_power_maps(bandPowerData, save_path):
    columns = 3
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(30, 20))
    for i in range(rows):
        for j in range(columns):
            bandName = bandPowerData.columns[rows * i + j]
            bandValues = bandPowerData.iloc[:, rows * i + j]
            mne.viz.plot_topomap(bandValues, raw.info, cmap=cm.inferno, contours=6,
                                 axes=axes[i][j], show=False, show_names=True, names=raw.ch_names)
            axes[i][j].set_title(f"{bandName} power", {"fontsize": 25})
    fig.savefig(save_path)


def plot_band_power_peaks(bandPower, save_path, fg):
    columns = 3
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(30, 20))
    for i in range(rows):
        for j in range(columns):
            bandName = bandPower.columns[rows * i + j]
            bandValues = bandPower.iloc[:, rows * i + j]
            argMax = np.argmax(bandValues)
            fg.get_fooof(argMax).plot(ax=axes[i][j], add_legend=False)
            axes[i][j].yaxis.set_ticklabels([])
            axes[i][j].set_title('biggest ' + bandName + ' peak', {'fontsize': 16})
            axes[i][j].set_title(f"{bandName} power", {"fontsize": 25})
    fig.savefig(save_path)


def print_error(msg):
    print('\x1b[0;31;40m' + str(msg) + '\x1b[0m')


if __name__ == '__main__':
    args = ParserPreparation().getArgs()
    sample_data_raw_file = args.input

    raw = mne.io.read_raw(sample_data_raw_file, preload=True, verbose=args.verbose)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    tmin = args.begin_time
    tmax = args.end_time
    fmin = args.frequency_min
    fmax = args.frequency_max

    raw.crop(tmin, tmax)
    raw.filter(fmin, fmax)

    if args.sfreq:
        raw.resample(sfreq=args.sfreq)

    nc = pyprep.NoisyChannels(raw)
    nc.find_all_bads()
    raw.info['bads'] = nc.get_bads()
    print(f"Bed electrods:{raw.info['bads']}")
    raw.interpolate_bads()
    bandPowerData = yasa.bandpower(raw)

    try:
        spectra, freqs = psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_overlap=150)
        bands = Bands({'theta': [3, 7],
                       'alpha': [7, 14],
                       'beta': [15, 30]})

        fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                        peak_threshold=2., max_n_peaks=6, verbose=False)
        freq_range = [fmin, fmax]
        fg.fit(freqs, spectra, freq_range)
        plot_band_power_peaks(bandPowerData, os.path.join(args.out_dir_path,
                                                          f"bandPowerPeaks.{args.chart_extension}"), fg)

    except ValueError as e:
        print_error(f"given range [{tmin}, {tmax}] is to short. Plot band power will not be made.")
        print_error(str(e))

    if args.verbose:
        data = raw.get_data(units="uV")
        sf = raw.info['sfreq']
        chan = raw.ch_names
        print('Chan =', chan)
        print('Sampling frequency =', sf, 'Hz')
        print('Data shape =', data.shape)

    if not os.path.exists(args.out_dir_path):
        os.makedirs(args.out_dir_path)

    plot_band_power_maps(bandPowerData, os.path.join(args.out_dir_path, f"bandPower.{args.chart_extension}"))

    bandPowerData.to_csv(os.path.join(args.out_dir_path, "bandPowers_raw.csv"))
    if args.convert:
        raw.save(os.path.join(args.out_dir_path, "result_fif.fif"), overwrite=True)

    pass
