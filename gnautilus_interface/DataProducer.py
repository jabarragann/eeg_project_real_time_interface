import traceback
import numpy as np
import pandas as pd
import time
from threading import Thread
import random
from tensorflow.keras.models import load_model
from pylsl import StreamInlet, resolve_stream
import yasa
from scipy.signal import welch


class DataProducer(Thread):

    def __init__(self, controller, data_queue):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.data_queue = data_queue

        #Connect to LSL stream
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])

        #LSTM model and configurations
        self.predictionModel = None
        self.timesteps = 2
        self.features = 50
        self.channels = 10
        self.dataBuffer = np.zeros((1500, self.channels))

        self.sequenceForPrediction = np.zeros((1, self.timesteps, self.features))

        #Load prediction model
        self.predictionModel = load_model('./models/modelNBack.h5')

        #Normalization Values from training
        normalizationValues = pd.read_csv('./models/modelNormalization.csv', sep=',')
        self.X_mean = normalizationValues['mean'].values
        self.X_std = normalizationValues['std'].values
        self.epsilon = 1e-8

        #Features Info
        self.sf = 250
        self.bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 16, 'Sigma'), (16, 30, 'Beta')]

        #Moving average filter
        self.alpha = 0.015
        self.init_value = 1

    def run(self):
        # Get Initial time
        self.init_time = time.time()

        # Collect Data until time session Time is over
        try:
            while self.controller.running:

                # Get new chunk of EEG Measurements
                chunk, timestamps = self.inlet.pull_chunk(timeout=2.5, max_samples=250)

                # Roll old data to make space for the new chunk
                self.dataBuffer = np.roll(self.dataBuffer, -250, axis=0)  # Buffer Shape (Time*SF, channels)
                self.dataBuffer[-250:, :] = np.array(chunk)  # Add chunk in the last 250 rows. 250 samples ==> 1 second.

                # Get bandPower coefficients
                win = int(1.8 * self.sf)  # Window size for Welch estimation.
                freqs, psd = welch(self.dataBuffer[-500:, :], self.sf, nperseg=win, axis=0)  # Calculate PSD on the last two seconds of data.

                # Calculate the bandpower on 3-D PSD array
                bandpower = yasa.bandpower_from_psd_ndarray(psd.transpose(), freqs,  self.bands)  # Bandpower shape ==> (#bands, #OfChannels)
                bandpower = bandpower.transpose().reshape(-1)

                # Add feature to vector to LSTM sequence and normalize sequence for prediction.
                self.sequenceForPrediction = np.roll(self.sequenceForPrediction, -1, axis=1)  # Sequence shape ==> (1, timesteps, #ofFeatures)
                self.sequenceForPrediction[0, -1, :] = bandpower  # Set new data point in last row
                normalSequence = (self.sequenceForPrediction - self.X_mean) / (self.X_std + 1e-8)  # Normalize sequence

                # Make Prediction
                prediction = self.predictionModel.predict(normalSequence)
                prediction = prediction[0, 1]
                predicted_label = 1 if prediction > 0.5 else 0

                # Apply moving average to the predicted labels.
                smooth_label = self.alpha * predicted_label + (1 - self.alpha) * self.init_value
                self.init_value = smooth_label

                t = time.time() - self.init_time

                # Only for simulation
                # predicted_label = self.sineSignal(t)

                # Send prediction for plotting
                print('time', t, 'prediction', prediction, 'label', predicted_label)
                self.data_queue.put_nowait({'time': t, 'data': predicted_label, 'smooth': smooth_label})

        except Exception as ex:
            print("Error!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            print("Closing data producer")

    @staticmethod
    def sineSignal(t):
        # return np.clip(np.sin(0.05*np.pi * t), -1,1)
        data_point = 1 if np.sin(0.1 * np.pi * t) > 0 else 0

        flip_label = 1 if random.uniform(0.0, 1.01) > 0.97 else 0

        if flip_label:
            if data_point == 1:
                data_point = 0
            elif data_point == 0:
                data_point = 1

        return data_point