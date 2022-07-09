import os
import mne
import pyprep
import numpy as np
import mffpy

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



    raw.plot(duration=5, n_channels=30)

    pass


