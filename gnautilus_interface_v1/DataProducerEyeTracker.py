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
from pylsl import StreamInfo, StreamOutlet,resolve_byprop
from gnautilus_interface_v1.sensor_utils.eye_tracker_utils import create_eye_tracker_features

class RecordEventInLSL:
    def __init__(self):
        self.pred_info = StreamInfo('PredictionEvents', 'predictions', 1, 0, 'float32', 'PredictionEvents43536')
        self.pred_outlet = StreamOutlet(self.pred_info)

        self.pred_smooth_info = StreamInfo('PredictionEventsSmooth', 'predictions', 1, 0, 'float32', 'PredictionEvents43536')
        self.pred_smooth_outlet = StreamOutlet(self.pred_smooth_info)

    def record_event(self, data, smooth_data):
        self.pred_outlet.push_sample([data])
        self.pred_outlet.push_sample([smooth_data])

class model_predictor:
    def __init__(self,):


        # Load model & normalizer
        user = 'multi01-real-time-exp' #'multiuser01-IU-real-time-exp'
        self.model = load_model('./deep_models_fuse_features/model_{:}_fuse.h5'.format(user))
        normalizer = pickle.load(open('./deep_models_fuse_features/normalizer_{:}_fuse.pic'.format(user), 'rb'))
        self.global_mean = normalizer['mean']
        self.global_std = normalizer['std']

    def make_prediction(self, fuse_sample):
        fuse_sample = fuse_sample.values
        # Normalize data
        test_x = (fuse_sample - self.global_mean) / self.global_std
        # Predict
        predictions = self.model.predict(test_x)
        predictions = predictions[0,0]

        # predicted_label = 1 if prediction > 0.5 else 0
        # print("prediction scores:", prediction, "label: ", predicted_label)

        return predictions

def create_eeg_features(eeg_epochs):
    eeg_epochs = eeg_epochs.transpose([0, 2, 1])
    # Calculate PSD
    sf = 250
    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(eeg_epochs, sf, nperseg=win, axis=-1)

    # Calculate bandpower
    bands = [(4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta')]
    bands_names = ['Theta', 'Alpha', 'Sigma', 'Beta']
    # Calculate the bandpower on 3-D PSD array
    bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=False)
    bandpower = bandpower.transpose([1, 2, 0])
    bandpower = bandpower.mean(axis=1)
    bandpower = pd.DataFrame(bandpower, columns=bands_names)

    return bandpower

class DataProducer(Thread):
    # USED_EEG_CHANNELS = ["FZ","PZ","CZ","C3","C4","CP1","CP2"]
    USED_EEG_CHANNELS = ["FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
                        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
                        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
                        "PZ", "P4", "P8", "PO3", "PO4"]
    CHANNEL_INDEX = {   "FP1":0 , "FP2":1, "AF3":2, "AF4":3, "F7":4, "F3":5, "FZ":6, "F4":7,
                        "F8":8, "FC5":9, "FC1":10, "FC2":11, "FC6":12, "T7":13, "C3":14, "CZ":15,
                        "C4":16, "T8":17, "CP5":18, "CP1":19, "CP2":20, "CP6":21, "P7":22, "P3":23,
                        "PZ":24, "P4":25, "P8":26, "PO7":27, "PO3":28, "PO4":29, "PO8":30, "OZ":31}
    GAZE_STREAM_INFO = {'gidx':0,'s':1,'gpx':2,'gpy':3}
    USED_GAZE = ['gpx','gpy']
    LEFT_EYE_STREAM_INFO = {'gidx':0,'s':1,'pcx':2,'pcy':3,'pcz':4,'pd':5,'gdx':6,'gdy':7,'gdz':8}
    USED_LEFT_EYE = ['pd']

    POWER_BANDS = ['Theta', 'Alpha', 'Beta']

    def __init__(self, controller, data_queue, alarm_deque):
        super().__init__()
        self.EEG_IDX = [self.CHANNEL_INDEX[ch] for ch in self.USED_EEG_CHANNELS]
        self.GAZE_IDX =  [self.GAZE_STREAM_INFO[ch] for ch in self.USED_GAZE]
        self.LEFT_EYE_IDX = [self.LEFT_EYE_STREAM_INFO[ch] for ch in self.USED_LEFT_EYE]

        self.init_time = time.time()
        self.controller = controller
        self.data_queue = data_queue
        self.alarm_deque = alarm_deque

        #Create model
        self.model = model_predictor()

        #Connect to LSL stream
        print("Waiting for sensor streams ...")
        self.streams = resolve_stream('type', 'eeg')
        self.eeg_inlet = StreamInlet(self.streams[0])
        print('eeg connected')
        self.streams = resolve_byprop('name', 'left_eye_data')
        self.left_eye_inlet = StreamInlet(self.streams[0])
        self.streams = resolve_byprop('name', 'gaze_position')
        self.gaze_inlet = StreamInlet(self.streams[0])
        print('eye tracker connected')

        #Moving average filter
        self.alpha = 0.015 * 10
        self.init_value = 1

        #Data buffers
        self.eeg_buffer = np.zeros((6350,len(self.EEG_IDX)))
        self.eeg_ts = np.zeros((6350,1))
        self.eye_tracker_buffer = np.zeros((2600,3))
        self.eye_tracker_ts = np.zeros((2600,1))

        #Data flow
        self.new_eeg_chunk_size = 250
        self.window_size = 15

        #LSL module
        self.lslSender = RecordEventInLSL()

    def run(self):
        # Get Initial time
        self.init_time = time.time()
        # Collect Data until time session Time is over
        try:
            #It is needed to pull data before the main loop
            eeg_chunk, eeg_new_ts = self.eeg_inlet.pull_chunk(timeout=4, max_samples=self.new_eeg_chunk_size)
            gaze_chunk, gaze_ts = self.gaze_inlet.pull_chunk(timeout=0.0)
            left_eye_chunk, left_eye_ts = self.left_eye_inlet.pull_chunk(timeout=0.0)

            #Init ts buffers
            self.eye_tracker_ts += eeg_new_ts[-1]
            self.eeg_ts += eeg_new_ts[-1]
            prev_eye_feat = pd.DataFrame(np.array([0.0,0.0,0.0,0.0,0.0]).reshape(1, 5), columns=["number_of_fix", "average_fix", "ssp", "nni_value", "pd"])

            while self.controller.running:
                #############
                #EEG sensor #
                #############
                # Get new chunk of EEG Measurements
                eeg_chunk, eeg_new_ts = self.eeg_inlet.pull_chunk(timeout=4, max_samples=self.new_eeg_chunk_size)
                eeg_chunk = np.array(eeg_chunk) # Should be of size (125,30) - (new_eeg_chunk_size, channels)
                eeg_chunk = eeg_chunk[:,self.EEG_IDX]
                # Roll old data to make space for the new chunk
                self.eeg_buffer = np.roll(self.eeg_buffer, -self.new_eeg_chunk_size, axis=0)
                self.eeg_ts = np.roll(self.eeg_ts, -self.new_eeg_chunk_size,axis=0)
                # Add chunk in the last rows of the data buffer. 250 samples ==> 1 second.
                self.eeg_buffer[-self.new_eeg_chunk_size:, :] = eeg_chunk[:, :]
                self.eeg_ts[-self.new_eeg_chunk_size:,0] = eeg_new_ts

                #############
                #Eye tracker#
                #############
                try:
                    # Get new chunk of eye tracker Measurements
                    gaze_chunk, gaze_ts = self.gaze_inlet.pull_chunk(timeout=0.0)
                    left_eye_chunk, left_eye_ts = self.left_eye_inlet.pull_chunk(timeout=0.0)
                    gaze_chunk, left_eye_chunk = np.array(gaze_chunk), np.array(left_eye_chunk)
                    gaze_chunk     = gaze_chunk[:, self.GAZE_IDX]
                    left_eye_chunk = left_eye_chunk[:, self.LEFT_EYE_IDX]
                    eye_tracker_chunk = np.hstack((gaze_chunk, left_eye_chunk)) #['gpx','gpy','pd']

                    # Roll old data to make space for the new chunk
                    eye_chunk_size = eye_tracker_chunk.shape[0]
                    self.eye_tracker_buffer = np.roll(self.eye_tracker_buffer, -eye_chunk_size, axis=0)
                    self.eye_tracker_ts = np.roll(self.eye_tracker_ts, -eye_chunk_size, axis=0)
                    self.eye_tracker_buffer[-eye_chunk_size:, :] = eye_tracker_chunk[:, :] # Add chunk in the last rows of the data buffer.
                    self.eye_tracker_ts[-eye_chunk_size:,0] = gaze_ts
                except Exception as e:
                    print("Error with eye tracker at ts {:}".format(self.eeg_ts[-1]))
                    traceback.print_exc()

                #Calculate indices for the last window_size seconds of data
                last_ts = self.eeg_ts[-1]
                eeg_window_idx = (self.eeg_ts > last_ts - self.window_size).squeeze().nonzero()
                eye_window_idx = (self.eye_tracker_ts > last_ts - self.window_size).squeeze().nonzero()

                eeg_window = self.eeg_buffer[eeg_window_idx,:]
                eeg_window_ts = self.eeg_ts[eeg_window_idx,:]
                eye_window = self.eye_tracker_buffer[eye_window_idx,:][0] #Data synchronized ready to use for feature calculation
                eye_window_ts = self.eye_tracker_ts[eye_window_idx,:][0]

                # Calculate features from buffers
                bandpower =  create_eeg_features(eeg_window)

                try:
                    eye_feat = create_eye_tracker_features(eye_window[:,0],eye_window[:,1],eye_window[:,2],eye_window_ts[:,0])
                    prev_eye_feat = eye_feat
                except:
                    eye_feat = prev_eye_feat
                    print("eye tracker feature calculation error")
                    traceback.print_exc()

                fuse_df = pd.DataFrame(np.hstack((eye_feat.values, bandpower.values)),columns=(list(eye_feat.columns) + list(bandpower.columns)))

                #debug
                assert eeg_chunk.shape[0] == 250, 'error with eeg'
                # print('eeg chunk shape', eeg_chunk.shape )
                # print('eye chunk shape', eye_tracker_chunk.shape)
                # print('eeg window shape', eeg_window.shape)
                # print('eye window shape', eye_window.shape)
                # print('eye window last ts', eye_window_ts[-1,0])
                # print(bandpower)
                # print(eye_feat)
                print(fuse_df.iloc[:,:5])
                print(fuse_df.iloc[:,5:])

                #Make predictions
                prediction = self.model.make_prediction(fuse_df)
                #Send prediction
                # Apply moving average to the predicted labels.
                smooth_label = self.alpha * prediction + (1 - self.alpha) * self.init_value
                self.init_value = smooth_label

                #Send data to lsl
                self.lslSender.record_event(prediction, smooth_label)

                #Send data to real-time interface
                t = time.time() - self.init_time

                # Send prediction for plotting
                self.data_queue.put_nowait({'time': t, 'data': prediction, 'smooth': smooth_label})

                #Update alarm deque
                self.alarm_deque.append(smooth_label)
                self.alarm_deque.popleft()

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