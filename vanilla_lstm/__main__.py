# generate a sequence of random numbers in [0, n_features)
from random import randint

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense


def generate_sequence(length, n_features):
    return [randint(0, n_features - 1) for _ in range(length)]


def one_hot_encode(sequence, n_features):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_features)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)


def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


def generate_example(length, n_features, out_index):
    sequence = generate_sequence(length, n_features)
    encoded = one_hot_encode(sequence, n_features)
    X = encoded.reshape((1, length, n_features))
    y = encoded[out_index].reshape(1, n_features)

    return X, y


if __name__ == '__main__':
    length = 5
    n_features = 10
    out_index = 2

    model = Sequential()
    model.add(LSTM(50, input_shape=(length, n_features)))
    model.add(Dense(n_features, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

# for i in range(10000):
#     X, y = generate_example(length, n_features, out_index)
#     yhat = model.fit(X, y, verbose=2)
