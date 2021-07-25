import Functions as func
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# reading in dataset
Main_data = pd.read_csv("Main_dataset.csv")
#print(Main_data.head(-1))


def remove_noise(sig):
    y, fs = func.plot_sig(sig)
    order = 6
    cutoff_lp = 200
    cutoff_hp = 25
    y = func.hp_filter(y, cutoff_hp, fs, order)
    y = func.lp_filter(y, cutoff_lp, fs, order)
    y_fft, xf = func.fourier_trans(y, fs)
    return y_fft


# create list to store fft of signals
x_train =[]
y_train =[]

# find the fourier transform of all the examples
for i in range(0, len(Main_data)):
    sig = Main_data["WAV"][i]
    cat = Main_data["CAT"][i]
    fft = remove_noise(sig)
    x_train.append(fft)
    y_train.append(cat)
    print(i, cat)

New_Heartbeat_Wav = pd.DataFrame(x_train)
New_Heartbeat_Wav["CATEGORY"] = y_train
print(New_Heartbeat_Wav.head(-1))

# let us encode the category column
label_encoder = preprocessing.LabelEncoder()

#creating list of features than needs to be encoded
encoded_features = ['CATEGORY']

# encode the list of features with numeric values
New_Heartbeat_Wav[encoded_features] = New_Heartbeat_Wav[encoded_features].apply(LabelEncoder().fit_transform)

# save the csv file
New_Heartbeat_Wav.to_csv("New_Wav_Heartbeat_Data.csv", index=False)
