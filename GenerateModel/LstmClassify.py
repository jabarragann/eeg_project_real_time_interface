from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GetData2 as datasetClass

def createModel1(timesteps,features):
    networkInput = Input(shape=(timesteps, features))

    dropout1 = Dropout(rate=0.4)(networkInput)
    hidden1 = Dense(12, activation='relu')(dropout1)
    batchNorm1 = BatchNormalization()(hidden1)

    hidden2 = LSTM(8, stateful=False)(batchNorm1)
    dropout2= Dropout(.1)(hidden2)
    hidden3 = Dense(2, activation='linear')(dropout2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1

def createModel2(timesteps,features):
    networkInput = Input(shape=(timesteps, features))

    dropout1 = Dropout(rate=0.1)(networkInput)
    hidden1 = Dense(6, activation='relu')(dropout1)
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

    dataset = datasetClass.EpocDataset('data', sessions=[1])

    trainX, trainY, testX, testY = dataset.getLstmData(['03','04'], sessArr=[1])


    print("Positive", "Negative")
    print("Train", len(np.where(trainY == 1)[0]), len(np.where(trainY == 0)[0]))
    print("Test1", len(np.where(testY == 1)[0]),len(np.where(testY == 0)[0]))

    #Normalize data
    X_mean = np.mean(trainX, axis=(0,1))*0.6
    X_std = np.std(trainX, axis=(0,1))*0.6

    #Save Normalization data
    normalizationDict = {'mean': X_mean,'std':X_std}
    normalizationFrame = pd.DataFrame(normalizationDict)
    normalizationFrame.to_csv('./Models/modelNBackNormalization.csv', index=False)

    trainX = (trainX - X_mean)/(X_std+epsilon)
    testX  = (testX - X_mean) /(X_std+epsilon)

    #Convert labels to one-hot encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    #Train
    timesteps = 16
    features= 70
    model = createModel2(timesteps,features)
    model.summary()

    #Optimal epochs=6
    history = model.fit(trainX,trainY, epochs=50, batch_size=512, validation_data=(testX,testY), shuffle=True)

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


