import os

import mne.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

import pyprep
from sklearn.decomposition import FastICA


def getFrame(file):
    return pd.read_table(file, header=None, delim_whitespace=True)


def savePlot1(frame, title, dirPath, fileName):
    '''
    A graph showing the signal cloud such that a sample from signal A corresponds to a sample in signal B
    '''
    plt.clf()
    X = frame.iloc[:, 0]
    Y = frame.iloc[:, 1]

    plt.title(title)
    plt.scatter(X, Y)
    filePath = os.path.join(dirPath, fileName)
    plt.savefig(filePath)


def savePlot2(frame, title, dirPath, fileName):
    plt.clf()
    t = np.arange(len(frame.index))
    for column in frame:
        s = frame[column]
        plt.plot(t, s)

    plt.title(title)
    filePath = os.path.join(dirPath, fileName)
    plt.savefig(filePath)


class EEGFastICA:

    def __init__(self, raw, args):
        self.raw = raw
        self.fun = args.fun
        self.raw_ica = None
        self.args = args

    def run(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#examples-using-sklearn-decomposition-fastica
        transformer = FastICA(random_state=97,
                              fun=self.fun,
                              whiten='unit-variance',
                              max_iter=50000000)
        self.data_before = pd.DataFrame(self.raw.get_data()[:args.n]).T
        self.date_after = pd.DataFrame(transformer.fit_transform(self.data_before))

    def run2(self):
        # https://mne.tools/stable/generated/mne.preprocessing.ICA.html
        import mne
        ica = mne.preprocessing.ICA(random_state=97)
        raw = self.raw.copy()
        ica.fit(raw)
        result = ica.apply(raw)
        self.raw_ica = raw.copy()



    def getResult(self):
        return self.raw_ica

    def saveResultToFile(self, filePath, header=False):
        if self.raw_ica is None:
            return
        self.raw_ica.to_csv(filePath, header=header)

    def saveChartsToFile(self, dirPath, fileName, N, M):
        # rows = raw.get_data()[:args.n]
        # frame = pd.DataFrame(rows, raw.ch_names[0:args.n])
        # frame_transpose = frame.T
        data_before = self.data_before
        data_after = self.date_after
        if N == 2:
            savePlot1(data_before, f"Before {N}x{M}", dirPath, "Before1_" + fileName)
            savePlot1(data_after, f"After {N}x{M}", dirPath, "After1_" + fileName)

        savePlot2(data_before, f"Before {N}x{M}", dirPath, "Before2_" + fileName)
        savePlot2(data_after, f"After {N}x{M}", dirPath, "After2_" + fileName)


class ParserPreparation:

    def __init__(self):
        _parser = argparse.ArgumentParser(description="Program for performing FastICA algorithm on dataframe. ")
        _parser.add_argument("-i", "--input", help="input file path", required=True, type=str)
        _parser.add_argument("-of", "--out-file-path", help="output file path. If exist will be overwritten.",
                             required=True, type=str)
        _parser.add_argument("-oc", "--out-chart-path", help="output directory path for charts",
                             required=False, type=str)
        _parser.add_argument("--header", type=bool, default=False, help="Bool flag if file has headers.")
        _parser.add_argument("-n", help="number of columns", required=True, type=int)
        _parser.add_argument("-m", help="number of rows", required=True, type=int)
        _parser.add_argument("-f", "--fun", required=False, default="logcosh",
                             choices=['logcosh', 'exp', 'cube'],
                             help="The functional form of the G function used in the approximation to neg-entropy")
        _parser.add_argument("-c", "--chart", help="Bool flag to decide whether to save the graphical chart",
                             default=True, required=False, type=bool)

        self._args = _parser.parse_args()

    def getArgs(self):
        return self._args


if __name__ == '__main__':
    argParser = ParserPreparation()
    args = argParser.getArgs()
    file = args.input
    fun = args.fun
    raw = mne.io.read_raw(file)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    raw.load_data()
    raw.crop(25, 35)
    raw.filter(1, 45)

    raw.resample(sfreq=64)
    nc = pyprep.NoisyChannels(raw)
    nc.find_all_bads()
    raw.info['bads'] = nc.get_bads()
    raw.interpolate_bads()


    eeg = EEGFastICA(raw, args)
    eeg.run()

    outResultFilePath = args.out_file_path
    outChartDir = args.out_chart_path
    eeg.saveResultToFile(outResultFilePath, args.header)

    # FileName without extension
    fileName = os.path.basename(args.input).split(".")[0]
    if args.chart:
        eeg.saveChartsToFile(outChartDir, fileName+f"_{fun}_" + ".png", args.n, args.m)

    # eeg.run()
    pass
