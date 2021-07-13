from tensorflow.keras.models import load_model
import numpy as np
import pickle

class simple_lstm_predictor:
    def __init__(self, lstm_model_name):
        #Load model and configuration
        self.model = load_model('./models/{:}.h5'.format(lstm_model_name))
        self.normalization_dict = pickle.load(open('./models/{:}_normalizer.pickle'.format(lstm_model_name), 'rb'))
        self.configuration_dict = pickle.load(open('./models/{:}_config.pickle'.format(lstm_model_name), 'rb'))

        self.sf =250
        self.window_length = self.configuration_dict['frame_length']
        self.overlap = self.configuration_dict['overlap']
        self.lstm_sequence_length = self.configuration_dict['sequence_length']

        self.window_size = int(self.sf*self.window_length)
        self.chunk_size  = int(self.sf*self.window_length - self.sf*self.overlap)

        self.dataBuffer = np.zeros((30000,30))
        self.sequenceForPrediction = np.zeros((1, self.lstm_sequence_length, 90))

        #Load prediction model and normalizer
        self.global_mean = self.normalization_dict['mean']
        self.global_std = self.normalization_dict['std']



new_model = simple_lstm_predictor("keyu_model_peg_knot/simple_lstm_seq20_keyu-peg-vs-knot-test-4-6").model
new_model.summary()