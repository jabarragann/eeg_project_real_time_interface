"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

from pylsl import StreamInlet, resolve_stream
import numpy as np
import time

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    beginTime = time.time()
    chunk, timestamps = inlet.pull_chunk(timeout=1.5, max_samples=250)
    if timestamps:
        print(timestamps, chunk)
        chunk = np.array(chunk)
        print(chunk.shape)
        endTime=time.time()
        print(endTime - beginTime)
        beginTime = time.time()