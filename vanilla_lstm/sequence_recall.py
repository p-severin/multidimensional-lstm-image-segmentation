from random import randint

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense


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

    for _ in range(5000):
        X, y = generate_example(length, n_features, out_index)
        model.train_on_batch(X, y)

    correct = 0
    for _ in range(100):
        X, y = generate_example(length, n_features, out_index)
        y_pred = model.predict(X, verbose=0)
        if one_hot_decode(y) == one_hot_decode(y_pred):
            correct += 1
    print(f'Accuracy: {correct}%')

    for _ in range(5):
        X, y = generate_example(length, n_features, out_index)
        y_pred = model.predict(X, verbose=0)
        print(f'Expected: {one_hot_decode(y)[0]}, Got: {one_hot_decode(y_pred)[0]}')
