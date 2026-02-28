import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense, Lambda, Softmax

from md_lstm.md_lstm_implementation import MultiDimensionalLSTM


def build_model(rows, cols, channels, hidden_size, num_classes):
    input_x = Input(shape=(rows, cols, channels), name='input_x')
    input_xv = Input(shape=(rows, cols, channels), name='input_xv')
    input_xh = Input(shape=(rows, cols, channels), name='input_xh')
    input_xvh = Input(shape=(rows, cols, channels), name='input_xvh')

    eps = Lambda(lambda t: t + 1e-4)
    x = eps(input_x)
    xv = eps(input_xv)
    xh = eps(input_xh)
    xvh = eps(input_xvh)

    # Layer 1: 4 directional MD-LSTM passes
    rnn_1 = MultiDimensionalLSTM(hidden_size, name='md_lstm_1_1')(x)
    rnn_1v = MultiDimensionalLSTM(hidden_size, name='md_lstm_1_2')(xv)
    rnn_1h = MultiDimensionalLSTM(hidden_size, name='md_lstm_1_3')(xh)
    rnn_1vh = MultiDimensionalLSTM(hidden_size, name='md_lstm_1_4')(xvh)

    merged = Concatenate(axis=-1)([rnn_1, rnn_1v, rnn_1h, rnn_1vh])
    intermediate = Dense(hidden_size, activation='tanh')(merged)

    # Flip intermediate features for layer 2 directional diversity
    flipped_v = Lambda(tf.image.flip_left_right)(intermediate)
    flipped_h = Lambda(tf.image.flip_up_down)(intermediate)
    flipped_vh = Lambda(lambda t: tf.image.flip_up_down(tf.image.flip_left_right(t)))(intermediate)

    # Layer 2: 4 directional MD-LSTM passes
    rnn_2 = MultiDimensionalLSTM(hidden_size, name='md_lstm_2_1')(intermediate)
    rnn_2v = MultiDimensionalLSTM(hidden_size, name='md_lstm_2_2')(flipped_v)
    rnn_2h = MultiDimensionalLSTM(hidden_size, name='md_lstm_2_3')(flipped_h)
    rnn_2vh = MultiDimensionalLSTM(hidden_size, name='md_lstm_2_4')(flipped_vh)

    merged_2 = Concatenate(axis=-1)([rnn_2, rnn_2v, rnn_2h, rnn_2vh])
    logits = Dense(num_classes)(merged_2)
    output = Softmax()(logits)

    model = Model(inputs=[input_x, input_xv, input_xh, input_xvh], outputs=output)
    model.summary()
    return model
