import os
import mne

#sample_audvis_filt-0-40_raw.fif
#sample_audvis_raw.fif
if __name__ == '__main__':
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                            'sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file)
    raw = raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')
    raw.load_data()
    raw.filter(1, 45)
    raw.crop(0,5)
    raw.resample(100)
    #raw.plot_psd(fmax=50)

    raw.plot(duration=5, n_channels=30)
    print(raw)
    print(raw.info)


    pass
