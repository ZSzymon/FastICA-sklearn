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

if __name__ == '__main__':
    sample_data_raw_file = "resources/1111.mff"

    raw = mne.io.read_raw(sample_data_raw_file,preload=True, verbose=True)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    tmin, tmax = 20, 30
    raw.crop(tmin, tmax)
    raw.filter(1, 45)
    raw.resample(sfreq=256)
    nc = pyprep.NoisyChannels(raw)
    nc.find_all_bads()
    raw.info['bads'] = nc.get_bads()
    print(f"Bed electrods:{raw.info['bads']}")
    raw.interpolate_bads()
    data = raw.get_data(units = "uV")
    sf = raw.info['sfreq']
    chan = raw.ch_names

    print('Chan =', chan)
    print('Sampling frequency =', sf, 'Hz')
    print('Data shape =', data.shape)
    bandPowerData = yasa.bandpower(raw)

    alphas = bandPowerData.iloc[:,0]
    columns = 3
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(15, 10))

    for i in range(rows):
        for j in range(columns):
            twoToOneDimensionIndex = rows*i + j
            bandName = bandPowerData.columns[rows*i + j]
            bandValues = bandPowerData.iloc[:, rows*i + j]
            mne.viz.plot_topomap(bandValues, raw.info, cmap=cm.inferno, contours=6,
                                 axes=axes[i][j], show=False)
            # yasa.topoplot(bandValues)

            axes[i][j].set_title(f"{bandName} power", {"fontsize": 20})

    fig.savefig("dest.eps", format="eps")
    pass


