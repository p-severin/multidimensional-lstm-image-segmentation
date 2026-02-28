import os

import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from md_lstm.models import build_model
from utils.data_generator import DataGenerator
from utils.pascal_voc import resolve_class_names

DATA_DIR = 'data/voc_2012_segmentation_data'

CLASS_WEIGHTS = [0.3, 0.7]


def class_weighted_pixelwise_crossentropy(target, output):
    output = K.clip(output, 1e-7, 1.0 - 1e-7)
    per_pixel_loss = -K.sum(target * K.log(output), axis=-1)
    pixel_weights = K.sum(target * CLASS_WEIGHTS, axis=-1)
    return per_pixel_loss * pixel_weights


class VisualizationCallback(Callback):
    def __init__(self, sample_batch, output_dir, save_interval=10):
        super().__init__()
        self.sample_x, self.sample_y = sample_batch
        self.output_dir = output_dir
        self.save_interval = save_interval
        self._current_epoch = 0
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.save_interval != 0:
            return

        preds = self.model.predict(self.sample_x, verbose=0)

        plt.subplot(151)
        plt.imshow(self.sample_x[0][0, :, :, :3])
        plt.title('original')
        plt.axis('off')

        plt.subplot(152)
        plt.imshow(preds[0, :, :, 1], cmap='Greys', vmin=0, vmax=1)
        plt.title('pred: person')
        plt.axis('off')

        plt.subplot(153)
        plt.imshow(self.sample_y[0, :, :, 1], cmap='Greys', vmin=0, vmax=1)
        plt.title('gt: person')
        plt.axis('off')

        plt.subplot(154)
        plt.imshow(preds[0, :, :, 0], cmap='Greys', vmin=0, vmax=1)
        plt.title('pred: bg')
        plt.axis('off')

        plt.subplot(155)
        plt.imshow(self.sample_y[0, :, :, 0], cmap='Greys', vmin=0, vmax=1)
        plt.title('gt: bg')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'image_{self._current_epoch}_{batch}.jpg'),
            bbox_inches='tight',
            dpi=100,
        )
        plt.close()


def train():
    dim = (32, 32)
    batch_size = 4
    epochs = 5
    hidden_size = 40
    num_classes = 2
    channels = 27

    chosen_classes = resolve_class_names(['person'])

    training_generator = DataGenerator(DATA_DIR, 'train', chosen_classes, batch_size=batch_size, dim=dim, shuffle=True)
    validation_generator = DataGenerator(DATA_DIR, 'val', chosen_classes, batch_size=batch_size, dim=dim, shuffle=False)

    model = build_model(dim[0], dim[1], channels, hidden_size, num_classes)

    optimizer = Adam(learning_rate=1e-3, clipnorm=5.0)
    model.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    models_dir = 'models/md_lstm'
    os.makedirs(models_dir, exist_ok=True)
    output_dir = 'output/md_lstm'

    sample_batch = training_generator[0]

    callbacks: list[Callback] = [
        ModelCheckpoint(
            os.path.join(models_dir, 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'), save_best_only=True
        ),
        VisualizationCallback(sample_batch, output_dir),
    ]

    history = model.fit(
        training_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
    )

    if history is None:
        raise ValueError('Model training failed')

    os.makedirs(output_dir, exist_ok=True)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()


if __name__ == '__main__':
    train()
