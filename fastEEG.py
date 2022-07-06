import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.decomposition import FastICA


def getFrame(file):
    return pd.read_table(file, header=None, delim_whitespace=True)


def savePlot1(frame, title, dirPath, fileName):
    '''
        Wykres prezentujący chmurę sygnału w taki sposób, że próbka z sygnału A odpowiada próbce w sygnale B
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

    def __init__(self, frame, fun="logcosh"):
        self.frame = frame
        self.fun = fun
        self.X_transformed_df = None

    def run(self):
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#examples-using-sklearn-decomposition-fastica
        transformer = FastICA(random_state=0,
                              fun=self.fun,
                              whiten='unit-variance')
        X_transformed = transformer.fit_transform(self.frame)
        self.X_transformed_df = pd.DataFrame(X_transformed)

    def getResult(self):
        return self.X_transformed_df

    def saveResultToFile(self, filePath, header=False):
        if self.X_transformed_df is None:
            return
        self.X_transformed_df.to_csv(filePath, header=header)

    def saveChartsToFile(self, dirPath, fileName, N, M):
        if N == 2:
            savePlot1(self.frame, f"Before {N}x{M}", dirPath, "Before1_" + fileName)
            savePlot1(self.X_transformed_df, f"After {N}x{M}", dirPath, "After1_" + fileName)

        savePlot2(self.frame, f"Before {N}x{M}", dirPath, "Before2_" + fileName)
        savePlot2(self.X_transformed_df, f"After {N}x{M}", dirPath, "After2_" + fileName)


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
    file = os.path.join(args.input)
    fun = args.fun
    eeg = EEGFastICA(getFrame(file), fun)
    eeg.run()
    outResultFilePath = args.out_file_path
    outChartDir = args.out_chart_path
    eeg.saveResultToFile(outResultFilePath, args.header)

    # FileName without extension
    fileName = os.path.basename(args.input).split(".")[0]
    if args.chart:
        eeg.saveChartsToFile(outChartDir, fileName + ".png", args.n, args.m)

    eeg.run()
    pass
