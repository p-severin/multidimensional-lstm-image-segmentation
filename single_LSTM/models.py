from keras import Input, Model
from keras import backend as K
from keras.layers import LSTM, Dense, Lambda, Permute, concatenate
from keras.optimizers import Adam
from keras.utils import plot_model

rows = 90
cols = 90
channels = 27
classes = 2
hidden_size = 40


def reshape_to_one_dimension(x):
    return K.reshape(x, (-1, cols, channels))


def reshape_to_one_dimension_hidden_size(x):
    return K.reshape(x, (-1, cols, hidden_size))


def reshape_to_two_dimensions(x):
    batch = K.shape(x)[0] // rows
    return K.reshape(x, (batch, rows, cols, -1))


def get_model_with_layer(path, layername):
    model = build_model(90, 90, 27)
    # model = load_model(path)
    optimizer = Adam(lr=10e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.load_weights(path)
    output_layer = model.get_layer(layername).output
    # output_layer = Lambda(lambda x: x[1, :, :, :])(output_layer)
    model = Model(inputs=model.input, outputs=output_layer)
    return model


def build_model(rows, cols, channels):
    input_x = Input(shape=(rows, cols, channels))
    x = Lambda(reshape_to_one_dimension)(input_x)
    x = LSTM(hidden_size, return_sequences=True)(x)
    x = Lambda(reshape_to_two_dimensions)(x)
    x = Permute((2, 1, 3))(x)
    x = Lambda(reshape_to_one_dimension_hidden_size)(x)
    x = LSTM(hidden_size, return_sequences=True)(x)
    x = Lambda(reshape_to_two_dimensions)(x)
    # x = Permute((1, 2, 3))(x)

    input_xv = Input(shape=(rows, cols, channels))
    xv = Lambda(reshape_to_one_dimension)(input_xv)
    xv = LSTM(hidden_size, return_sequences=True)(xv)
    xv = Lambda(reshape_to_two_dimensions)(xv)
    xv = Permute((2, 1, 3))(xv)
    xv = Lambda(reshape_to_one_dimension_hidden_size)(xv)
    xv = LSTM(hidden_size, return_sequences=True)(xv)
    xv = Lambda(reshape_to_two_dimensions)(xv)
    # xv = Permute((1, 2, 3))(xv)

    input_xh = Input(shape=(rows, cols, channels))
    xh = Lambda(reshape_to_one_dimension)(input_xh)
    xh = LSTM(hidden_size, return_sequences=True)(xh)
    xh = Lambda(reshape_to_two_dimensions)(xh)
    xh = Permute((2, 1, 3))(xh)
    xh = Lambda(reshape_to_one_dimension_hidden_size)(xh)
    xh = LSTM(hidden_size, return_sequences=True)(xh)
    xh = Lambda(reshape_to_two_dimensions)(xh)
    # xh = Permute((1, 2, 3))(xh)

    input_xvh = Input(shape=(rows, cols, channels))
    xvh = Lambda(reshape_to_one_dimension)(input_xvh)
    xvh = LSTM(hidden_size, return_sequences=True)(xvh)
    xvh = Lambda(reshape_to_two_dimensions)(xvh)
    xvh = Permute((2, 1, 3))(xvh)
    xvh = Lambda(reshape_to_one_dimension_hidden_size)(xvh)
    xvh = LSTM(hidden_size, return_sequences=True)(xvh)
    xvh = Lambda(reshape_to_two_dimensions)(xvh)
    # xvh = Permute((1, 2, 3))(xvh)

    merge_layer = concatenate([x, xv, xh, xvh])
    # batch_norm = BatchNormalization()(dense_before)
    dense = Dense(classes, activation='softmax')(merge_layer)

    model = Model(inputs=[input_x, input_xv, input_xh, input_xvh], outputs=[dense])
    model.summary()
    return model


if __name__ == '__main__':
    model = build_model(rows, cols, channels)
    # optimizer = Adam(lr=10e-4)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    plot_model(model)

    # model = get_model_with_layer(path, layername=)

    # callbacks = [ModelCheckpoint(os.path.join('/home/pseweryn/Projects/multidimensional_lstm/repository/models', 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'),
    #                              save_best_only=True)]

    # directory_voc_dataset = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'

    # dataset = Dataset(directory_voc_dataset, subsets=['train', 'val'], image_shape=(rows * 3, cols * 3))
    # X, y = dataset.create_patches('train', how_many_images=-1)
    # print(X.shape)
    # print(y.shape)
    # print('dataset created')
    # model.fit(x=[X[:, 0], X[:, 1], X[:, 2], X[:, 3]], y=y, batch_size=batch_size, epochs=5, verbose=1)
