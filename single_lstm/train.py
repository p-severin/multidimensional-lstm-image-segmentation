import os

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from single_lstm.custom_loss_function import class_weighted_pixelwise_crossentropy
from single_lstm.data_generator import DataGenerator
from single_lstm.models import build_model

DATA_DIR = 'data/voc_2012_segmentation_data'

if __name__ == '__main__':
    dim = (90, 90)
    chosen_classes = [15]

    training_generator = DataGenerator(DATA_DIR, 'train', chosen_classes, batch_size=4, dim=dim, shuffle=True)
    validation_generator = DataGenerator(DATA_DIR, 'val', chosen_classes, batch_size=4, dim=dim, shuffle=False)

    model = build_model(dim[0], dim[1], 27)
    model.summary()
    optimizer = Adam(learning_rate=10e-4)
    model.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    folder_to_save_models = 'models/single_lstm'
    if not os.path.exists(folder_to_save_models):
        os.makedirs(folder_to_save_models)

    callbacks: list[Callback] = [
        ModelCheckpoint(
            os.path.join(folder_to_save_models, 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'), save_best_only=True
        )
    ]

    history = model.fit(
        training_generator,
        epochs=10,
        callbacks=callbacks,
        validation_data=validation_generator,
    )

    if history is None:
        raise ValueError('Model training failed')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./accuracy.png')
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./loss.png')
    plt.close()
