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
sns.set(style='white', font_scale=1.2)

from scipy.signal import welch


waves = {
    "Gamma": (30, 45),
    "Beta": (13, 30),
    "Alpha": (7, 12),
    "Theta": (4, 7),
    "Delta": (1, 4),
}
if __name__ == '__main__':
    sample_data_raw_file = "resources/1111.mff"

    raw = mne.io.read_raw(sample_data_raw_file,preload=True, verbose=True)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    tmin, tmax = 20, 60
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


    win = int(4*sf)
    freqs, psd = welch(data, sf, nperseg=win, average='median')

    print(freqs.shape, psd.shape)

    plt.plot(freqs, psd[1], 'k', lw=2)
    plt.fill_between(freqs, psd[1], cmap='Spectral')
    plt.xlim(0, 50)
    plt.yscale('log')
    sns.despine()
    plt.title(chan[1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')

    bandPowerData = yasa.bandpower(raw)
    from matplotlib import cm, colors, colorbar
    alphas = bandPowerData.iloc[:,1]
    mne.viz.plot_topomap(bandPowerData.iloc[:,1], raw.info, cmap= cm.viridis, contours=0)

    pass


