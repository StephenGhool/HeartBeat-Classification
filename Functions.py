import numpy
import numpy as np
import librosa
import librosa.display
import pandas as pd
from matplotlib import pyplot as plt
import librosa
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
import librosa.display
import noisereduce as nr
import pywt as pw

# reading in dataset
Main_data = pd.read_csv("Main_dataset.csv")


# plotting the signal in the time domain
def plot_sig(wav_path):
    # creating a figure to plot the data
    # figure = plt.figure(figsize=(14, 6))
    # load the wav file
    audio_type, sample_rate = librosa.load(wav_path)
    # librosa.display.waveplot(audio_type, sr=sample_rate)
    # plt.show()
    return audio_type, sample_rate

# function for plotting fourier transform
def fourier_trans(data, sample_rate):
    normalized = data
    N = len(data)
    yf = fft(normalized)
    xf = fftfreq(N, 1/sample_rate)
    # plt.plot(xf[:2000], np.abs(yf[:2000]))
    # plt.show()
    # print(N, len(yf), len(xf))
    return np.abs(yf[:2000]), xf[:2000]


# to remove some of the noise we could apply a low pass to remove unwanted frequencies in the data
def lp_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# creating a high pass filter to remove lower frequencies
def hp_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

# we are using fft...let us only focus on one side as well as, the frequency from 0 to 200Hz...ignore all other values
# def fft_reduce(fft):
#     n = len(fft)
#     fft_half = fft[int(n/2):n]
#     return

# # testing the plot function
# sig = Main_data["WAV"][3]
# y, fs = plot_sig(sig)
#
# # testing fourier transform
# fourier_trans(y)
#
# # testing low pass filter
# order = 6
# cutoff_lp = 200
# cutoff_hp = 20
# y = hp_filter(y,cutoff_hp,fs,order)
# y = lp_filter(y,cutoff_lp,fs,order)
# # # testing fourier transform
# yf, xf =fourier_trans(y,fs)
# fft_reduce(yf)
# print (xf)
# #
# librosa.display.waveplot(y, sr=fs)
# plt.show()
