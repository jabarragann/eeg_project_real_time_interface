import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import re

class Dataset:



    def __init__(self, timesteps, dataPath):

        if dataPath is not None:
            self.dataPath = dataPath
        else:
            self.dataPath = Path("./freq-data/").resolve()

        self.timesteps = timesteps

        self.trainX, self.trainY = self.getData(self.dataPath / "train")
        self.valX, self.valY = self.getData(self.dataPath / "val")
        self.testX, self.testY = self.getData(self.dataPath / "test")
        x = 0


    def getData(self, dataPath):
        X = []
        Y = []
        for file in dataPath.rglob('*.pickle'):

            task = re.findall("(?<=_S[0-9]_T[0-9]_).+(?=\.pickle)", file.name)[0]
            label = 0.0 if task == "Baseline" else 1.0

            with open(file,'rb') as fh:
                data = pickle.load(fh)
                hasNan = np.isnan(data).any()

                assert not hasNan, "NAN values in the data matrix of {:}".format(file.name)


            X += [data]
            Y += [np.ones(data.shape[0])*label]

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        features = X.shape[1]

        X,Y = self.series_to_supervised(X,Y,n_in=self.timesteps - 1)
        X = X.values.astype('float32')

        samples = X.shape[0]
        X = X.reshape((samples, self.timesteps, features))

        return X,Y

    @staticmethod
    def series_to_supervised(data,labels, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        labels = labels[n_in:]
        return agg, labels

if __name__ == '__main__':

    dataset = Dataset(timesteps=2)


