from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Permute, Reshape

rows = 96
cols = 96
channels = 3
classes = 21


def build_model():
    model = Sequential()
    model.add(Reshape((rows * cols, channels)))
    model.add(LSTM(30, return_sequences=True))
    model.add(Reshape((rows, cols, 30)))
    model.add(Permute((2, 1, 3)))
    model.add(Reshape((rows * cols, 30)))
    model.add(Bidirectional(LSTM(30, return_sequences=True)))
    model.add(Dense(classes, activation='softmax'))
    model.add(Reshape((rows, cols, classes)))
    model.summary()
    return model
