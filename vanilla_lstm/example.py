import random

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed


def get_sequence(n_timesteps):
    X = np.array([random.random() for _ in range(n_timesteps)])

    limit = n_timesteps / 4.0

    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])

    return X, y


def get_sequences(n_sequences, n_timesteps):
    seqX, seqY = list(), list()

    for _ in range(n_sequences):
        X, y = get_sequence(n_timesteps)
        seqX.append(X)
        seqY.append(y)

    seqX = np.array(seqX).reshape(n_sequences, n_timesteps, 1)
    seqY = np.array(seqY).reshape(n_sequences, n_timesteps, 1)
    return seqX, seqY


n_timesteps = 10

model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

X, y = get_sequences(50000, n_timesteps)
model.fit(X, y, epochs=1, batch_size=10)

X, y = get_sequences(100, n_timesteps)
loss, acc = model.evaluate(X, y, verbose=0)
print(f'Loss : {loss}, Accuracy: {acc * 100}')

for _ in range(10):
    X, y = get_sequences(1, n_timesteps)
    y_pred = (model.predict(X, verbose=0) > 0.5).astype('int32')
    exp, pred = y.reshape(n_timesteps), y_pred.reshape(n_timesteps)
    print(f'y = {y}, y_pred = {y_pred}, correct = {np.array_equal(exp, pred)}')
