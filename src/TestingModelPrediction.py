#Deep Learning Modules
from tensorflow.keras.models import Model, load_model
import json
import numpy as np
import pandas as pd

if __name__ == '__main__':

    epsilon = 1e-8
    predictionModel = load_model('./Model/modelEyes.h5')

    normalizationValues = pd.read_csv('./Model/modelEyesNormalization.csv', delimiter=',')
    X_mean = normalizationValues['mean'].values
    X_std = normalizationValues['std'].values


    with open('./Model/singleDataLine.txt','r') as f1:
        data = f1.readline()
        data = json.loads(data)

        data = np.array(data['pow'])
        data = data.reshape(1,-1)
        print(data.shape)
        print(predictionModel.summary())

        #Normalize
        data = (data - X_mean) / (X_std + epsilon)
        prediction = predictionModel.predict(data)

        print(prediction)
        print(prediction[0,1])