from gnautilus_interface_v1.deep_models import LoadData
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def createModel2(timesteps,features):
    networkInput = Input(shape=(timesteps, features))

    dropout1 = Dropout(rate=0.1)(networkInput)
    hidden1 = Dense(30, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    hidden2 = LSTM(6, stateful=False, dropout=0.1)(batchNorm1)
    hidden3 = Dense(2, activation='linear')(hidden2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1


if __name__ == '__main__':
    epsilon =1e-8
    timesteps = 2
    features = 150
    # dataset = LoadData.Dataset(timesteps=timesteps, dataPath =Path("./freq-data/only10ChannelsSmallWindows"))
    dataset = LoadData.Dataset(timesteps=timesteps, dataPath =Path("./freq-data/allChannelsSmallWindows"))

    trainX, trainY, testX, testY = dataset.trainX, dataset.trainY, dataset.testX, dataset.testY

    print("Positive", "Negative")
    print("Train", len(np.where(trainY == 1)[0]), len(np.where(trainY == 0)[0]))
    print("Test1", len(np.where(testY == 1)[0]),len(np.where(testY == 0)[0]))

    #Normalize data
    globalMean = trainX.mean(axis=(0, 1))
    globalStd = trainX.std(axis=(0, 1))
    trainX = (trainX - globalMean) / (globalStd + 1e-18)
    testX = (testX - globalMean) / (globalStd + 1e-18)

    #Save Normalization data
    normalizationDict = {'mean': globalMean,'std':globalStd}
    normalizationFrame = pd.DataFrame(normalizationDict)
    normalizationFrame.to_csv('./models/modelNormalization.csv', index=False)

    #Convert labels to one-hot encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    #Train
    model = createModel2(timesteps,features)
    model.summary()

    #Optimal epochs=6
    history = model.fit(trainX,trainY, epochs=250, batch_size=512, validation_data=(testX,testY), shuffle=True)

    #Print Max accuracy
    print("Training max accuracy: {:0.6f}".format(max(history.history['acc'])))
    print("Testing max accuracy:  {:0.6f}".format(max(history.history['val_acc'])))

    # Save Model
    print("Saved model to disk")
    model.save("./Models/modelNBack.h5")

    #Plot accuracy
    fig, axes = plt.subplots(2,1,sharex=True)
    axes[0].plot(history.history['acc'], label='train acc')
    axes[0].plot(history.history['val_acc'], label='test acc')
    axes[1].plot(history.history['loss'], label='train loss')
    axes[1].plot(history.history['val_loss'], label='test loss')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    y_test_hat_t = model.predict(testX, verbose=0)
    y_test_hat = np.argmax(y_test_hat_t, axis=1)
    y_test = np.argmax(testY, axis=1)

    fig, axes = plt.subplots(4,1,sharex=True)
    axes[0].stem(y_test, use_line_collection=True)
    axes[1].stem(y_test_hat, use_line_collection=True)
    axes[2].stem(y_test_hat_t[:,0], use_line_collection=True)
    axes[3].stem(y_test_hat_t[:,1], use_line_collection=True)
    plt.show()


