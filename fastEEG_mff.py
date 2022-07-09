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
    plt.scatter(X, Y, 1)
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
        self.args = args
        self.result = None
    def run(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#examples-using-sklearn-decomposition-fastica
        transformer = FastICA(random_state=97,
                              fun=self.fun,
                              whiten='unit-variance',
                              max_iter=50000000)
        self.data_before = pd.DataFrame(self.raw.get_data()[:args.n]).T
        self.date_after = pd.DataFrame(transformer.fit_transform(self.data_before))
        self.result = self.date_after



    def getResult(self):
        return self.result

    def saveResultToFile(self, filePath, header=False):
        if self.result is None:
            return
        self.result.to_csv(filePath, header=header)

    def saveChartsToFile(self, dirPath, fileName, N, M):
        data_before = self.data_before
        data_after = self.date_after
        if N == 2:
            savePlot1(data_before, f"Before {N}x{M} with function {self.fun}", dirPath, "Before1_mff" + fileName)
            savePlot1(data_after, f"After {N}x{M} with function {self.fun} ", dirPath, "After1_" + fileName)

        savePlot2(data_before, f"Before {N}x{M} with function {self.fun} ", dirPath, "Before2_mff" + fileName)
        savePlot2(data_after, f"After {N}x{M} with function {self.fun} ", dirPath, "After2_mff" + fileName)


class ParserPreparation:

    def __init__(self):
        _parser = argparse.ArgumentParser(description="Program for performing FastICA algorithm on dataframe. ")
        _parser.add_argument("-i", "--input", help="input file path", required=True, type=str)
        _parser.add_argument("-of", "--out-file-path", help="output file path. If exist will be overwritten.",
                             required=True, type=str)
        _parser.add_argument("-oc", "--out-chart-path", help="output directory path for charts",
                             required=False, type=str)
        _parser.add_argument("--header", type=bool, default=False,  help="Bool flag if file has headers.")
        _parser.add_argument("-n", help="number of columns", required=True, type=int)
        _parser.add_argument("-m", help="number of rows", required=True, type=int)
        _parser.add_argument("-f", "--fun", required=False, default="logcosh",
                             choices=['logcosh', 'exp', 'cube'],
                             help="The functional form of the G function used in the approximation to neg-entropy")
        _parser.add_argument("-c", "--chart", help="Bool flag to decide whether to save the graphical chart",
                             default=True, required=False, type=bool)

        _parser.add_argument("-b", "--begin-time", help="Start time of the raw data to use in seconds (must be >= 0).",
                             required=False, type=int, default=25, const=25, nargs='?')
        _parser.add_argument("-e", "--end-time", help="End time of the raw data to use in seconds (cannot exceed "
                                                      "data duration).", default=65, const=65, nargs='?',
                             required=False, type=int)
        _parser.add_argument("-s", "--sfreq", help="New sample rate to use.",
                             required=False, type=int, default=None, const=64, nargs='?')

        _parser.add_argument("-fmin", "--frequency-min", help="Minimum value of filtering frequency",
                             required=False, type=int, default=1, const=1, nargs='?')
        _parser.add_argument("-fmax", "--frequency-max", help="Max value of filtering frequency",
                             required=False, type=int, default=45, const=45, nargs='?')

        _parser.add_argument("-v", "--verbose", help="Control verbosity of the logging output. If None, use the "
                                                     "default verbosity level.", type=bool, default=None,
                             nargs='?', required=False, const=True)
        _parser.add_argument('-ce', '--chart-extension', help="Chart extension",
                             type=str, default="eps", const="eps", nargs='?')


        self._args = _parser.parse_args()

    def getArgs(self):
        return self._args


def read_raw_file(args):
    file = args.input
    raw = mne.io.read_raw(file)
    raw = raw.pick_types(meg=False, eeg=True, eog=False)
    raw.load_data()
    return raw


def cropData(raw, args):
    tmin = args.begin_time
    tmax = args.end_time
    raw.crop(tmin, tmax)
    return raw

def filterData(raw, args):
    fmin = args.frequency_min
    fmax = args.frequency_max
    raw.filter(fmin, fmax)
    return raw


def cleanData(raw):
    nc = pyprep.NoisyChannels(raw)
    nc.find_all_bads()
    raw.info['bads'] = nc.get_bads()
    raw.interpolate_bads()
    return raw


def performICA(raw):
    eeg = EEGFastICA(raw, args)
    eeg.run()
    return raw, eeg


if __name__ == '__main__':
    argParser = ParserPreparation()
    args = argParser.getArgs()
    raw = read_raw_file(args)
    raw = cropData(raw, args)
    raw = filterData(raw, args)

    if args.sfreq:
        raw.resample(sfreq=args.sfreq)

    raw = cleanData(raw)

    raw, eeg = performICA(raw)

    outChartDir = args.out_chart_path
    if not os.path.exists(outChartDir):
        os.makedirs(outChartDir)

    outResultFilePath = args.out_file_path
    eeg.saveResultToFile(outResultFilePath, args.header)

    # FileName without extension
    fileName = os.path.basename(args.input).split(".")[0]

    if args.chart:
        eeg.saveChartsToFile(outChartDir, fileName + f"_{args.fun}_" + ".png", args.n, args.m)

    # eeg.run()
    pass
