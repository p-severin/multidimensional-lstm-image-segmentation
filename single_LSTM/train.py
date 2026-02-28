import os

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from single_LSTM.custom_loss_function import class_weighted_pixelwise_crossentropy
from single_LSTM.models import build_model
from utils.images import Dataset

if __name__ == '__main__':
    directory_voc_dataset = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'

    # training_generator = DataGenerator(directory_voc_dataset, 'train', batch_size=4, shuffle=True, n_channels=27, dim=(90, 90))
    # validation_generator = DataGenerator(directory_voc_dataset, 'val', batch_size=4, shuffle=False, n_channels=27, dim=(90, 90))

    model = build_model(90, 90, 27)
    model.summary()
    optimizer = Adam(lr=10e-4)
    model.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    folder_to_save_models = (
        '/home/pseweryn/Projects/multidimensional_lstm/repository/models/one_class_only_without_permute'
    )
    if not os.path.exists(folder_to_save_models):
        os.makedirs(folder_to_save_models)
    callbacks = [
        ModelCheckpoint(
            os.path.join(folder_to_save_models, 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'), save_best_only=True
        )
    ]
    # model.fit_generator(training_generator, epochs=5, verbose=1, callbacks=callbacks,
    #                     validation_data=validation_generator)

    training_dataset = Dataset(directory_voc_dataset, 'train', [15])
    validation_dataset = Dataset(directory_voc_dataset, 'val', [15])
    X_train, y_train = training_dataset.generate_data(500)
    X_val, y_val = validation_dataset.generate_data(100)
    history = model.fit(
        x=[X_train[0], X_train[1], X_train[2], X_train[3]],
        y=y_train,
        batch_size=1,
        epochs=10,
        callbacks=callbacks,
        validation_data=[[X_val[0], X_val[1], X_val[2], X_val[3]], y_val],
    )

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./accuracy.png')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./loss.png')
    plt.show()
