"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import yasa
from scipy.signal import welch
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

plt.yscale('log')

#Normalization values
normalizationValues = pd.read_csv('./../models/modelNormalization.csv', sep=',')
X_mean = normalizationValues['mean'].values
X_std = normalizationValues['std'].values

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

dataBuffer = np.zeros((1500,10))
sequenceForPrediction = np.zeros((1, 2, 50))

sf = 250
bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
         (12, 16, 'Sigma'), (16, 30, 'Beta')]

predictionModel = load_model('./../models/modelNBack.h5')

try:
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        beginTime = time.time()
        chunk, timestamps = inlet.pull_chunk(timeout=1.5, max_samples=250)
        if timestamps:
            # print(timestamps, chunk)
            chunk = np.array(chunk)
            print(chunk.shape)
            endTime=time.time()
            print(endTime - beginTime)
            beginTime = time.time()

            #Roll old data to make space for the new chunk
            dataBuffer = np.roll(dataBuffer, -250, axis=0) #Buffer Shape (Time*SF, channels)
            dataBuffer[-250:, :] = np.array(chunk) #Add chunk in the last 250 rows. 250 samples ==> 1 second.

            # Get bandPower coefficients
            win = int(1.8 * sf)  # Window size for Welch estimation.
            freqs, psd = welch(dataBuffer[-500:, :], sf, nperseg=win, axis=0) #Calculate PSD on the last two seconds of data.

            #Calculate the bandpower on 3-D PSD array
            bandpower = yasa.bandpower_from_psd_ndarray(psd.transpose(), freqs, bands) #Bandpower shape ==> (#bands, #OfChannels)
            bandpower = bandpower.transpose().reshape(-1)

            #Add feature to vector to LSTM sequence and normalize sequence for prediction.
            sequenceForPrediction = np.roll(sequenceForPrediction, -1, axis=1) #Sequence shape ==> (1, timesteps, #ofFeatures)
            sequenceForPrediction[0, -1, :] = bandpower  #Set new data point in last row
            normalSequence = (sequenceForPrediction - X_mean) / (X_std + 1e-8) #Normalize sequence

            pred = predictionModel.predict(normalSequence)
            label = pred[0, 1]
            label = 1 if label > 0.5 else 0

            print(pred, label)
            x=0
finally:
    pass
    # for idx, i in enumerate(dataBuffer[:,0]):
    #     print(idx, i)