import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import time
from threading import Thread
import random
from tensorflow.keras.models import load_model
from pylsl import StreamInlet, resolve_stream
import yasa
from scipy.signal import welch
import pickle
from pylsl import StreamInfo, StreamOutlet

class RecordEventInLSL:

    def __init__(self):
        self.pred_info = StreamInfo('PredictionEvents', 'predictions', 1, 0, 'float32', 'PredictionEvents43536')
        self.pred_outlet = StreamOutlet(self.pred_info)
    def record_event(self, data):
        self.pred_outlet.push_sample([data])

class lstm_predictor:
    def __init__(self, model_path, lstm_model_name, EEG_CHANNELS, POWER_BANDS):
        self.lslSender = RecordEventInLSL()

        #Load model and configuration
        self.predictionModel = load_model(model_path / '{:}.h5'.format(lstm_model_name))
        self.normalization_dict = pickle.load(open(model_path / '{:}_normalizer.pickle'.format(lstm_model_name), 'rb'))
        self.configuration_dict = pickle.load(open(model_path / '{:}_config.pickle'.format(lstm_model_name), 'rb'))

        self.sf =250
        self.window_length = self.configuration_dict['frame_length']
        self.overlap = self.configuration_dict['overlap']
        self.lstm_sequence_length = self.configuration_dict['sequence_length']

        self.window_size = int(self.sf*self.window_length)
        self.chunk_size  = int(self.sf*self.window_length - self.sf*self.overlap)

        self.dataBuffer = np.zeros((30000,30))+1.2
        self.sequenceForPrediction = np.zeros((1, self.lstm_sequence_length, 90))

        #Load prediction model and normalizer
        self.global_mean = self.normalization_dict['mean']
        self.global_std = self.normalization_dict['std']

        #Channels and power bands
        self.EEG_CHANNELS = EEG_CHANNELS
        self.POWER_BANDS = POWER_BANDS

        print("Deep model config")
        print("sf {:d} window length {:0.3f} overlap {:0.3f}"
              " lstm length {:d} Window size {:d} Chunk size {:d}".format(  self.sf,
                                                                            self.window_length,
                                                                            self.overlap,
                                                                            self.lstm_sequence_length,
                                                                            self.window_size,
                                                                            self.chunk_size))

    def make_prediction(self, chunk):

        ###################
        ## Get new samples#
        ###################
        chunk = np.array(chunk) #Should be of size (125,30)
        # Roll old data to make space for the new chunk
        self.dataBuffer = np.roll(self.dataBuffer, -self.chunk_size, axis=0)  # Buffer Shape (Time*SF, channels)
        # Add chunk in the last 125 rows of the data buffer. 250 samples ==> 1 second.
        self.dataBuffer[-self.chunk_size:, :] = chunk[:, :] # Add chunk in the last 125 rows. 250 samples ==> 1 second.

        ######################
        ##Calculate Features##
        ######################
        # Check that data is in the correct range
        # check_units = [0.2 < abs(self.dataBuffer[-2*self.chunk_size:, 2].min()) < 800,
        #                 1 < abs(self.dataBuffer[-2*self.chunk_size:, 7].max()) < 800,
        #                 0.2 < abs(self.dataBuffer[-2*self.chunk_size:, 15].min()) < 800]
        # assert all(check_units), \
        #     "Check the units of the data that is about to be process. " \
        #     "Data should be given as uv to the get bandpower coefficients function" \
        #     + str(check_units)

        # Get bandPower coefficients
        win_sec = 0.95
        bandpower = yasa.bandpower(self.dataBuffer[-2*self.chunk_size:, :].transpose(),
                                   sf=self.sf, ch_names=self.EEG_CHANNELS, win_sec=win_sec,
                                   bands=[(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 40, 'Beta')])
        bandpower = bandpower[self.POWER_BANDS].transpose()
        bandpower = bandpower.values.reshape(1, -1)

        # Add feature to vector to LSTM sequence and normalize sequence for prediction.
        self.sequenceForPrediction = np.roll(self.sequenceForPrediction, -1, axis=1) #Sequence shape (1, timesteps, #features)
        self.sequenceForPrediction[0, -1, :] = bandpower  # Set new data point in last row

        #normalize sequence
        normalSequence = (self.sequenceForPrediction - self.global_mean)/self.global_std

        prediction = self.predictionModel.predict(normalSequence)
        prediction = prediction[0, 1]

        # predicted_label = 1 if prediction > 0.5 else 0

        # print("prediction scores:", prediction, "label: ", predicted_label)
        self.lslSender.record_event(prediction)

        return prediction

class DataProducer(Thread):
    USED_EEG_CHANNELS = ["FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
                        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
                        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
                        "PZ", "P4", "P8", "PO3", "PO4", "OZ"]
    CHANNEL_INDEX = {   "FP1":0 , "FP2":1, "AF3":2, "AF4":3, "F7":4, "F3":5, "FZ":6, "F4":7,
                        "F8":8, "FC5":9, "FC1":10, "FC2":11, "FC6":12, "T7":13, "C3":14, "CZ":15,
                        "C4":16, "T8":17, "CP5":18, "CP1":19, "CP2":20, "CP6":21, "P7":22, "P3":23,
                        "PZ":24, "P4":25, "P8":26, "PO7":27, "PO3":28, "PO4":29, "PO8":30, "OZ":31}

    POWER_BANDS = ['Theta', 'Alpha', 'Beta']

    def __init__(self, controller, data_queue, alarm_deque):
        super().__init__()
        self.init_time = time.time()
        self.controller = controller
        self.data_queue = data_queue
        self.alarm_deque = alarm_deque

        #Connect to LSL stream
        self.streams = resolve_stream('type', 'eeg')
        self.inlet = StreamInlet(self.streams[0])

        self.USED_INDEX = [self.CHANNEL_INDEX[ch] for ch in self.USED_EEG_CHANNELS]

        # #LSTM model and configurations
        # self.predictor = lstm_predictor(Path('./models/eyes_detector_model'),'simple_lstm_seq5_eyes', self.USED_EEG_CHANNELS,self.POWER_BANDS)
        # self.predictor = lstm_predictor(Path('./models/jing_model_peg_knot'),'simple_lstm_seq15_peg-vs-knot-test-1-3', self.USED_EEG_CHANNELS,self.POWER_BANDS)
        # self.predictor = lstm_predictor(Path('./models/juan_model_peg_knot'), 'simple_lstm_seq20_juan-peg-vs-knot-test-4-6', self.USED_EEG_CHANNELS, self.POWER_BANDS)
        # self.predictor = lstm_predictor(Path('./models/keyu_model_peg_knot'), 'simple_lstm_seq20_keyu-peg-vs-knot-test-4-6', self.USED_EEG_CHANNELS, self.POWER_BANDS)
        # self.predictor = lstm_predictor(Path('./models/ben_model_peg_knot'), 'simple_lstm_seq20_ben-peg-vs-knot-test-4-6', self.USED_EEG_CHANNELS, self.POWER_BANDS)
        # self.predictor = lstm_predictor(Path('./models/jing_model_needle_blood'), 'simple_lstm_seq25_Jing-needle-vs-needleBleeding-test-4-6', self.USED_EEG_CHANNELS, self.POWER_BANDS)
        # self.predictor = lstm_predictor(Path('./models/juan_model_needle_grasping'), 'simple_lstm_seq10_Juan-needle-vs-needlegrasping-test-4-6', self.USED_EEG_CHANNELS, self.POWER_BANDS)
        self.predictor = lstm_predictor(Path('./models/juan_model_needle_blood_last_try'), 'simple_lstm_seq25_Juan-needle-vs-needleBlood-last-try', self.USED_EEG_CHANNELS, self.POWER_BANDS)

        # self.predictionModel = None
        # self.timesteps = 2
        # self.features = 150
        # self.channels = 30
        # self.dataBuffer = np.zeros((1500, self.channels))
        #
        # self.sequenceForPrediction = np.zeros((1, self.timesteps, self.features))
        #
        # #Load prediction model
        # self.predictionModel = load_model('models/model30channels/modelNBack.h5')
        #
        # #Normalization Values from training
        # normalizationValues = pd.read_csv('models/model30channels/modelNormalization.csv', sep=',')
        # self.X_mean = normalizationValues['mean'].values
        # self.X_std = normalizationValues['std'].values
        # self.epsilon = 1e-8
        #
        # #Features Info
        # self.sf = 250
        # self.bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
        #          (12, 16, 'Sigma'), (16, 30, 'Beta')]

        self.chunk_size = self.predictor.chunk_size
        #Moving average filter
        self.alpha = 0.015 * 10
        self.init_value = 1

    def run(self):
        # Get Initial time
        self.init_time = time.time()

        # Collect Data until time session Time is over
        try:
            while self.controller.running:

                # Get new chunk of EEG Measurements
                chunk, timestamps = self.inlet.pull_chunk(timeout=4, max_samples=self.chunk_size)
                chunk = np.array(chunk)
                chunk = chunk[:,self.USED_INDEX]

                # # Roll old data to make space for the new chunk
                # self.dataBuffer = np.roll(self.dataBuffer, -250, axis=0)  # Buffer Shape (Time*SF, channels)
                # chunk = np.array(chunk)  # Add chunk in the last 250 rows. 250 samples ==> 1 second.
                # chunk = chunk[:, :-1] * 1e-6
                # self.dataBuffer[-250:, :] = chunk # Add chunk in the last 250 rows. 250 samples ==> 1 second.
                #
                # # Get bandPower coefficients
                # win = int(1.8 * self.sf)  # Window size for Welch estimation.
                # freqs, psd = welch(self.dataBuffer[-500:, :], self.sf, nperseg=win, axis=0)  # Calculate PSD on the last two seconds of data.
                #
                # # Calculate the bandpower on 3-D PSD array
                # bandpower = yasa.bandpower_from_psd_ndarray(psd.transpose(), freqs,  self.bands)  # Bandpower shape ==> (#bands, #OfChannels)
                # bandpower = bandpower.reshape(-1)
                #
                # # Add feature to vector to LSTM sequence and normalize sequence for prediction.
                # self.sequenceForPrediction = np.roll(self.sequenceForPrediction, -1, axis=1)  # Sequence shape ==> (1, timesteps, #ofFeatures)
                # self.sequenceForPrediction[0, -1, :] = bandpower  # Set new data point in last row
                # normalSequence = (self.sequenceForPrediction - self.X_mean) / (self.X_std + 1e-8)  # Normalize sequence
                #
                # # Make Prediction
                # prediction = self.predictionModel.predict(normalSequence)
                # prediction = prediction[0, 1]
                # predicted_label = 1 if prediction > 0.5 else 0

                prediction = self.predictor.make_prediction(chunk)
                predicted_label = 1 if prediction > 0.5 else 0

                # Apply moving average to the predicted labels.
                smooth_label = self.alpha * prediction + (1 - self.alpha) * self.init_value
                self.init_value = smooth_label

                t = time.time() - self.init_time

                # Only for simulation
                # predicted_label = self.sineSignal(t)

                # Send prediction for plotting
                # print('time', t, 'prediction', prediction, 'label', predicted_label)
                self.data_queue.put_nowait({'time': t, 'data': prediction, 'smooth': smooth_label})

                #Update alarm deque
                self.alarm_deque.append(smooth_label)
                self.alarm_deque.popleft()
                # print(list(self.alarm_deque))

        except Exception as ex:
            print("Error!!")
            print(ex)
            print(traceback.format_exc())
        finally:
            self.controller.running=False
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