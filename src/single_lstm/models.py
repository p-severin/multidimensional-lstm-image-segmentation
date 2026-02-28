from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Permute, TimeDistributed, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

rows = 90
cols = 90
channels = 27
classes = 2
hidden_size = 40


def get_model_with_layer(path, layername):
    model = build_model(90, 90, 27)
    optimizer = Adam(lr=10e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.load_weights(path)
    output_layer = model.get_layer(layername).output
    model = Model(inputs=model.input, outputs=output_layer)
    return model


def _build_branch(input_layer):
    x = TimeDistributed(LSTM(hidden_size, return_sequences=True))(input_layer)
    x = Permute((2, 1, 3))(x)
    x = TimeDistributed(LSTM(hidden_size, return_sequences=True))(x)
    x = Permute((2, 1, 3))(x)
    return x


def build_model(rows, cols, channels):
    input_x = Input(shape=(rows, cols, channels))
    input_xv = Input(shape=(rows, cols, channels))
    input_xh = Input(shape=(rows, cols, channels))
    input_xvh = Input(shape=(rows, cols, channels))

    x = _build_branch(input_x)
    xv = _build_branch(input_xv)
    xh = _build_branch(input_xh)
    xvh = _build_branch(input_xvh)

    merge_layer = concatenate([x, xv, xh, xvh])
    dense = Dense(classes, activation='softmax')(merge_layer)

    model = Model(inputs=[input_x, input_xv, input_xh, input_xvh], outputs=[dense])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model(rows, cols, channels)
    plot_model(model)
