import time
import numpy as np
import mne
from pylsl import StreamInfo, StreamOutlet
from pathlib import Path

def renameChannels(chName):
    if 'Z' in chName:
        chName = chName.replace('Z','z')
    if 'P' in chName and 'F' in chName:
        chName = chName.replace('P','p')
    return chName


# # next make an outlet
# channelsSubset = ['FZ', 'F7', 'F3', 'F4', 'F8', 'T8','T7','P7','P8','OZ']
# info = StreamInfo('Gnautilus', 'EEG',10, 250, 'float32', 'myuid34234')
# outlet = StreamOutlet(info)

if __name__ == '__main__':

    #Read eeg file
    srcPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\edf\JuanValidation")
    file = srcPath / "UJuan_S01_T01_BloodValidation_raw.edf"

    # srcPath = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\TestsWithVideo\Eyes-open-close-test\edf').resolve()
    # file = srcPath / "UJuan_S03_T01_Eyes-open-close_raw.edf"
    # srcPath = Path('./../deep_models/raw-data').resolve()
    # file = srcPath / "Juan_S04_T01_Low_raw.edf"
    # srcPath = Path(r'C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Jing\11-17-20\edf').resolve()
    # file = srcPath / "UJing_S03_T03_VeinLigationBlood_raw.edf"
    #Load data
    raw = mne.io.read_raw_edf(file, preload=True)
    rawArray = raw.get_data(picks=['eeg'])

    # Create LSL outlet
    info = StreamInfo("Gnautilus recording", 'eeg', 32, 250, 'float32', "gnaut.recorded")
    # Append channel meta-data
    info.desc().append_child_value("manufacturer", "G.tec")
    channels = info.desc().append_child("channels")
    for c in raw.ch_names:
        channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")

    outlet = StreamOutlet(info)
    print("Sending data ...")
    for idx in range(rawArray.shape[1]):
        # now send it and wait for a bit
        outlet.push_sample(rawArray[:,idx]*1e6)
        time.sleep(0.004)

    print("Finished sending data")