import numpy as np
import pandas as pd
import os
import re
import random
from scipy.stats import median_absolute_deviation
import matplotlib.pyplot as plt

class EpocDataset:
    def __init__(self, subject, sessions = []):

        print("Getting data class for v3_data")
        self.subject = subject
        self.path = './{:}/'.format(subject)
        self.X = np.array([])
        self.y = np.array([])

        self.trials = {}
        self.dataDictHigh = {}
        self.dataDictLow = {}

        self.highKeys = []
        self.lowKeys = []
        self.dataDict = {}

        self.sessions = sessions

        for sess in sessions:
            srcPath = self.path + 'D{:d}/'.format(sess, self.subject)
            self.trials[sess] = []
            self.dataDictHigh[sess] = []
            self.dataDictLow[sess] = []
            self.dataDict[sess]={}

            for f1 in os.listdir(srcPath):
                print(srcPath+f1)

                #Get trial
                trial =  re.findall("(?<=T)[0-9]{1,2}(?=_)", f1)[0]
                label = re.findall("(?<=T[0-9][0-9]_).+(?=_pow)", f1)[0]
                # print(label)
                data = pd.read_csv(srcPath+f1, delimiter=',')

                #Remove samples that don't have a label
                data = data[5:]
                tempIdx = 0
                while data['Label'].values[tempIdx]==-1:
                    tempIdx += 1

                data = data[tempIdx+1:-3]

                if label == 'High':
                    self.dataDictHigh[sess] += [[data,trial]]
                    self.highKeys += [trial]
                elif label == "Low":
                    self.dataDictLow[sess] += [[data, trial]]
                    self.lowKeys += [trial]

                self.dataDict[sess][trial] = data
                self.trials[sess].append(trial)
                # print(data.shape[0])


    def getData(self):
        print(self.trials)
        # self.trials = ['1','2','3','4']
        for sess in self.sessions:
            count = 0
            random.shuffle(self.dataDictHigh[sess])
            random.shuffle(self.dataDictLow[sess])

            orderHigh = [trial for data, trial in self.dataDictHigh[sess]]
            orderLow = [trial for data, trial in self.dataDictLow[sess]]

            print('orderHigh',orderHigh)
            print('orderLow', orderLow)
            for t in self.trials[sess]:
                dataHigh = self.dataDictHigh[sess][count][0]
                dataLow  = self.dataDictLow[sess][count][0]

                data = np.concatenate((dataHigh.values[:,2:],dataLow.values[:,2:]))
                labels = np.concatenate((dataHigh.values[:, 1],dataLow.values[:, 1]))
                self.X = np.concatenate((self.X, data)) if self.X.size else data
                self.y = np.concatenate((self.y, labels)) if self.y.size else labels

                count += 1
                if count >= len(self.dataDictHigh[sess]):
                    break

        return self.X, self.y

    def getTrainTest(self, testKeys,removeKeys=[], sess =1):
        testX = []
        testY = []
        trainX = []
        trainY = []

        for trial, _  in self.dataDict[sess].items():
            data = self.dataDict[sess][trial].values[:, 2:]
            label = self.dataDict[sess][trial].values[:, 1]
            if trial in removeKeys:
                pass
            elif trial in testKeys:
                testX = np.concatenate((testX, data)) if len(testX) else data
                testY = np.concatenate((testY, label)) if len(testY) else label
            else:
                trainX = np.concatenate((trainX, data)) if len(trainX) else data
                trainY = np.concatenate((trainY, label)) if len(trainY) else label

        X = np.concatenate((trainX,testX))
        y = np.concatenate((trainY, testY))

        return X, y

    def getLstmData(self, testKeys,removeKeys=[], sessArr =[1]):
        testX = []
        testY = []
        trainX = []
        trainY = []
        timesteps = 16

        average_mad = np.zeros(70)
        count = 0

        for sess in sessArr:
            for trial, _ in self.dataDict[sess].items():
                count += 1
                data = self.dataDict[sess][trial].values[:, 2:]
                label = self.dataDict[sess][trial].values[:, 1]

                # Clip data bigger than 3 standard deviations
                data_mad = median_absolute_deviation(data, axis=0)
                average_mad = average_mad + data_mad

                data = np.clip(data, np.zeros_like(data_mad),  data_mad * 4.0)

                lstmData, label = self.series_to_supervised(data,label, n_in=timesteps-1)
                lstmData = lstmData.values.astype('float32')

                features = data.shape[1]
                samples = lstmData.shape[0]

                lstmData = lstmData.reshape((samples, timesteps, features))

                if trial in removeKeys:
                    pass
                elif trial in testKeys:
                    testX = np.concatenate((testX, lstmData)) if len(testX) else lstmData
                    testY = np.concatenate((testY, label)) if len(testY) else label
                else:
                    trainX = np.concatenate((trainX, lstmData)) if len(trainX) else lstmData
                    trainY = np.concatenate((trainY, label)) if len(trainY) else label

        #Save Average Mad
        # print("Saving average mad")
        # average_mad = average_mad/count
        # average_mad = {'average_mad': average_mad,'average_mad_2': average_mad}
        # average_mad = pd.DataFrame.from_dict(average_mad)
        # average_mad.to_csv('./Models/modelNBackNormalization.csv', index=False)

        return trainX, trainY, testX, testY

    def series_to_supervised(self, data,labels, n_in=1, n_out=1, dropnan=True):
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

    def getNormalizationValues(self, sessArr=[1]):
        trainX = []
        for sess in sessArr:
            for trial, sessionData in self.dataDict[sess].items():
                data = sessionData.values[:, 2:]
                label = sessionData.values[:, 1]

                trainX = np.concatenate((trainX, data)) if len(trainX) else data

        # Clip data bigger than 4 median absolute deviations deviations
        data_mad = median_absolute_deviation(data, axis=0)
        trainX = np.clip(trainX, np.zeros_like(data_mad), data_mad * 4.0)

        #Get Normalization values after clipping
        X_mean = np.mean(trainX, axis=0) * 0.6
        X_median = np.median(trainX, axis=0) * 0.6
        X_std = np.std(trainX, axis=0) * 0.6
        X_mad =  median_absolute_deviation(trainX, axis=0) * 0.6

        statisticsDict = {'mean':X_mean,'median':X_median,'std':X_std,'mad':X_mad}
        dataFrame = pd.DataFrame.from_dict(statisticsDict)
        dataFrame.to_csv('./Models/NBackNormalization.csv', index=False)

        return X_mean, X_std, X_mad

    def getLstmData2(self, testKeys, removeKeys=[], sessArr=[1], testing=False, testing_mad=[]):
        testX = []
        testY = []
        trainX = []
        trainY = []
        timesteps = 16
        epsilon = 1e-8

        if not testing:
            X_mean, X_std, X_mad = self.getNormalizationValues(sessArr)
        else:
            X_mad = testing_mad

        for sess in sessArr:
            for trial, _ in self.dataDict[sess].items():
                data = self.dataDict[sess][trial].values[:, 2:]
                label = self.dataDict[sess][trial].values[:, 1]

                # Clip data bigger than 4 standard deviations
                data = np.clip(data, np.zeros_like(X_mad), X_mad * 4.0)

                lstmData, label = self.series_to_supervised(data, label, n_in=timesteps - 1)
                lstmData = lstmData.values.astype('float32')

                features = data.shape[1]
                samples = lstmData.shape[0]

                lstmData = lstmData.reshape((samples, timesteps, features))
                lstmData = (lstmData - X_mean) / (X_std + epsilon)

                if trial in removeKeys:
                    pass
                elif trial in testKeys:
                    testX = np.concatenate((testX, lstmData)) if len(testX) else lstmData
                    testY = np.concatenate((testY, label)) if len(testY) else label
                else:
                    trainX = np.concatenate((trainX, lstmData)) if len(trainX) else lstmData
                    trainY = np.concatenate((trainY, label)) if len(trainY) else label

        return trainX, trainY, testX, testY


if __name__ == "__main__":

    dataset = EpocDataset('Juan', sessions=[1,2,3])

    X, y = dataset.getData()
    print(np.where(y == -1)[0])
    print(X.shape[0])

    X,y =  dataset.getTrainTest(['03','08'], removeKeys=['02','05'])

    print(X.shape)
    print(y.shape)


    trainX, trainY, testX, testY = dataset.getLstmData(['01','02'], sessArr=[2,3])
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    # frame =  dataset.getNormalizationValues(sessArr=[1,2])
    # data =  dataset.getLstmData2(['01','02'], sessArr=[1,2])