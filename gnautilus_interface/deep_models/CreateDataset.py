import numpy as np
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yasa
from scipy.signal import welch
from pathlib import Path
import pickle


def createEvenlySpaceEvents(totalPoints,w1,sf, ):
    ##Software is removing last epoch of data
    ##Solution create events manually
    eTime = int(w1 / 2 * sf)
    events = [[eTime, 0, 1]]
    while eTime < totalPoints:
        eTime += int(sf * w1 * 0.5)
        events.append([eTime, 0, 1])

    events = np.array(events)

    return events

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName


channelsSubset = ['FZ', 'F7', 'F3', 'F4', 'F8', 'T8','T7','P7','P8','OZ']

if __name__ == '__main__':

    srcPath =  Path('./raw-data/').resolve()
    dstPath  =  Path('./freq-data/').resolve()

    for file in srcPath.glob("*.edf"):
        #Read eeg file
        # file = srcPath / "exp2_Chiho_S2_T2_Low.edf"
        print(file.name)
        raw = mne.io.read_raw_edf(file, preload=True)
        raw.drop_channels(['PO7','PO8'])
        raw = raw.pick(channelsSubset)
        rawArray = raw.get_data(picks=['eeg'])

        #Rename Channel
        mne.rename_channels(raw.info, renameChannels)
        #Set montage (3d electrode location)
        raw = raw.set_montage('standard_1020')

        #Create events every w1 seconds
        sf = 250
        w1 = 2
        events_array = createEvenlySpaceEvents(rawArray.shape[1], w1, sf)
        #Create Epochs
        epochs = mne.Epochs(raw, events_array, tmin=-(w1 / 2 - 0.02 * w1), tmax=(w1 / 2 - 0.02 * w1))

        #Get data as numpy array
        epochsArray = epochs.get_data(picks=['eeg'])
        numbOfEpochs = epochsArray.shape[0]

        win = int(1.8 * sf)  # Window size is set to 4 seconds
        freqs, psd = welch(epochsArray, sf, nperseg=win, axis=-1)

        # Calculate the bandpower on 3-D PSD array
        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 16, 'Sigma'), (16, 30, 'Beta')]
        bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands)

        bandpower = bandpower.transpose([1,0,2]).reshape((numbOfEpochs,-1))

        #Shape should be (#ofEpochs, #OfChannels*#ofbands)
        print(bandpower.shape)

        with open(dstPath / file.with_suffix(".pickle").name, 'wb') as outF:
            pickle.dump(bandpower,outF)

        x = 0

