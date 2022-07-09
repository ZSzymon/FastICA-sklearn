import argparse
import os
import mne
import pyprep
import numpy as np
import mffpy
import matplotlib as plt
import mne
import yasa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap
from mne.time_frequency import psd_welch

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum
from matplotlib import cm, colors, colorbar


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    '''
    https://stackoverflow.com/questions/12151306/argparse-way-to-include-default-values-in-help
    '''

    def add_argument(self, *args, help=None, default=None, **kwargs):
        if help is not None:
            kwargs['help'] = help
        if default is not None and args[0] != '-h':
            kwargs['default'] = default
            if help is not None:
                kwargs['help'] += ' Default: {}'.format(default)
        super().add_argument(*args, **kwargs)


class ParserPreparation:

    def __init__(self):
        parser = argparse.ArgumentParser(description="Program for visualization EEG data in different band power.",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        required = parser.add_argument_group('Required arguments')
        optional = parser.add_argument_group('Optional arguments')

        required.add_argument("-i", "--input", help="input file path", required=True, type=str)
        required.add_argument("-o", "--out-file-path", help="output file path. If exist will be overwritten.",
                              required=True, type=str)
        optional.add_argument("-b", "--begin-time", help="Start time of the raw data to use in seconds (must be >= 0).",
                              required=False, type=int, default=25, const=25, nargs='?')
        optional.add_argument("-e", "--end-time", help="End time of the raw data to use in seconds (cannot exceed "
                                                       "data duration).",
                              required=False, type=int)
        optional.add_argument("-v", "--verbose", help="Control verbosity of the logging output. If None, use the "
                                                      "default verbosity level.", type=bool, default=False,
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


def plot_band_power_peaks(bandPower, save_path):
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


if __name__ == '__main__':
    args = ParserPreparation().getArgs()
    sample_data_raw_file = args.input

    raw = mne.io.read_raw(sample_data_raw_file, preload=True, verbose=args.verbose)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    tmin = args.begin_time
    tmax = args.end_time
    raw.crop(tmin, tmax)
    raw.filter(1, 45)
    raw.resample(sfreq=256)
    nc = pyprep.NoisyChannels(raw)
    nc.find_all_bads()
    raw.info['bads'] = nc.get_bads()
    print(f"Bed electrods:{raw.info['bads']}")
    raw.interpolate_bads()


    spectra, freqs = psd_welch(raw, fmin=1, fmax=40, tmin=0, tmax=250,
                               n_overlap=150, n_fft=300)
    bands = Bands({'theta': [3, 7],
                   'alpha': [7, 14],
                   'beta': [15, 30]})

    fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                    peak_threshold=2., max_n_peaks=6, verbose=False)
    freq_range = [1, 45]
    fg.fit(freqs, spectra, freq_range)

    if args.verbose:
        data = raw.get_data(units="uV")
        sf = raw.info['sfreq']
        chan = raw.ch_names
        print('Chan =', chan)
        print('Sampling frequency =', sf, 'Hz')
        print('Data shape =', data.shape)

    bandPowerData = yasa.bandpower(raw)
    plot_band_power_maps(bandPowerData, args.out_file_path)
    plot_band_power_peaks(bandPowerData, args.out_file_path.split(".")[0]+ "2." + args.out_file_path.split(".")[1])

    pass
