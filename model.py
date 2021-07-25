import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# loading dataset
New_Heartbeat_Wav = pd.read_csv("New_Wav_Heartbeat_Data.csv")
print(New_Heartbeat_Wav.head(-1))
print(New_Heartbeat_Wav["CATEGORY"].value_counts())

# split dataset into X and y
X = New_Heartbeat_Wav.iloc[:, :-1]
y = New_Heartbeat_Wav["CATEGORY"].values
# print(X)
# print(y)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.7, random_state=42, shuffle=True)
print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)

#scale the model
scaler = Normalizer()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)

# converting all to arrays to be reshaped
xTrain = pd.DataFrame(xTrain)
xTrain = xTrain.values
xTest = pd.DataFrame(xTest)
xTest = xTest.values

# reshaping the arrays
xTrain = xTrain.reshape(409, 2000, 1, 1)
xTest = xTest.reshape(176, 2000, 1, 1)
yTrain = yTrain.reshape(409, 1)
yTest = yTest.reshape(176, 1)

print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)

# MODEL USING SKLEARN SVM CLASSIFICATION

# model = svm.SVC(decision_function_shape='ovo')
# model.fit(xTrain, yTrain)
# y_pred = model.predict(xTest)
#
# #Model Performance
# print("Mean Sq Err: %.2f" % mean_squared_error(yTest,y_pred))
# print("accuracy: ", model.score(xTrain, yTrain))
# print("accuracy: ", model.score(xTest, yTest))

# WE NEED TO USE NEURAL NETWORKS
# sequential model using keras
model = keras.Sequential(
    [
        keras.Input(shape=(2000, 1)),
        layers.Conv1D(64, 10, padding='valid', activation="relu"),
        layers.MaxPool1D(pool_size=3),
        layers.Conv1D(64, 2, activation="relu"),
        layers.MaxPool1D(),
        layers.Conv1D(128, 2, activation="relu"),
        layers.MaxPool1D(),
        layers.Conv1D(256, 2, activation="relu"),
        layers.MaxPool1D(),
        layers.Conv1D(512, 2, activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(1024, activation="relu"),
        layers.Dense(5)
    ]
)
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)
model.fit(xTrain, yTrain, batch_size=64, epochs=10, verbose=2)
model.evaluate(xTest, yTest, batch_size=64, verbose=2)
