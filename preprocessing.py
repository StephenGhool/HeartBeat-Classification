import Functions as func
import pandas as pd
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import noisereduce as nr

# reading in dataset
input_data = pd.read_csv("New_Wav_Heartbeat_Data.csv")

# # remove the cat column to process data
cat = input_data["CATEGORY"]
input_data["CATEGORY"] = input_data["1550"]
data = input_data
print(data)

# convert the dataframe to an array
data_arr = data.values

# standard parameters
max_bpm = 200
fs = 86
peaks = []
beats = []
reduced_noise =[]

# iterate to remove noise, find number of peaks, distance between peaks
for i in range(len(data)):
    signal = data_arr[i]
    print(signal)
    r_peaks = func.QRS_detection(signal, fs, max_bpm)
    beats.append(len(r_peaks))
    peaks.append(r_peaks)
    # reduced_noise.append(nr.reduce_noise(y=signal, sr=fs))

# create a dataframe to hold beats, peaks distance, category
features = pd.DataFrame()
features = pd.DataFrame(peaks)
features =features.fillna(0)
features["Beats"] = pd.DataFrame(beats)
features["CAT"] = pd.DataFrame(cat)

print(features)

# save the data to a CSV file
features.to_csv("Features.csv",index=False)