import os

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

from classes_pascal import pascal_ids
from single_lstm.models import build_model, get_model_with_layer
from utils.images import Dataset

plt.rcParams.update({'font.size': 6})

validation_data_dir = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'
save_directory = '/home/pseweryn/Projects/multidimensional_lstm/repository/results'
experiment_name = 'saving_layers'

segmentation_true_dir = 'segmentation_true'
segmentation_predicted_dir = 'segmentation_predicted'
original_dir = 'original'

save_segmentation_true_dir = os.path.join(save_directory, experiment_name, segmentation_true_dir)
save_segmentation_predicted_dir = os.path.join(save_directory, experiment_name, segmentation_predicted_dir)
save_original_dir = os.path.join(save_directory, experiment_name, original_dir)
save_layers_dir = os.path.join(save_directory, experiment_name, 'layers')

for dir in [save_layers_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

batch_size = 4
rows = 90
cols = 90
nrows = 5
ncols = 8


def validate(model: Model, batch_size, layer):
    """Train the model.
    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
    """

    dataset = Dataset(validation_data_dir, 'val', chosen_classes=[15], image_shape=(270, 270))
    X, y = dataset.generate_data(100)
    print(X.shape)

    y_pred = model.predict([X[0], X[1], X[2], X[3]], batch_size=batch_size, verbose=1)

    print(y_pred.shape)

    for i in range(y_pred.shape[0]):
        # X = np.expand_dims(X, axis=0)
        # y_pred = model.predict([np.expand_dims(X[:, 0], 0),
        #                         np.expand_dims(X[:, 1], 0),
        #                         np.expand_dims(X[:, 2], 0),
        #                         np.expand_dims(X[:, 3], 0)], batch_size=batch_size, verbose=1)
        # y_pred = np.argmax(y_pred, axis=3)

        # y_true = np.argmax(y[i], axis=2)

        # plt.subplot(151)
        # plt.imshow(dataset.original_images[i])
        # plt.axis('off')
        # plt.title('oryginał')
        #
        # plt.subplot(152)
        # plt.imshow(dataset.y[i, :, :, 0], cmap='Greys')
        # plt.axis('off')
        # plt.title('segmentacja: tło')
        #
        # plt.subplot(153)
        # plt.imshow(y_pred[i, :, :, 0], cmap='Greys')
        # plt.axis('off')
        # plt.title('predykcja: tło')
        #
        # plt.subplot(154)
        # plt.imshow(dataset.y[i, :, :, 1], cmap='Greys')
        # plt.axis('off')
        # plt.title('segmentacja: człowiek')
        #
        # plt.subplot(155)
        # plt.imshow(y_pred[i, :, :, 1], cmap='Greys')
        # plt.axis('off')
        # plt.title('predykcja: człowiek')
        #
        # plt.tight_layout()

        # plt.show()
        # plt.savefig(
        #     '/home/pseweryn/Projects/multidimensional_lstm/repository/results/person/image_{}.jpg'.format(i), bbox_inches='tight', dpi=100)
        # plt.close()
        # plt.savefig(
        #     os.path.join(save_original_dir, 'original_iteration_{}.jpg'.format(i)))

        fig, ax = plt.subplots(nrows, ncols)
        fig.set_size_inches((8, 2), forward=False)
        for row in range(nrows):
            for col in range(ncols):
                ax[col].imshow(y[i, :, :, col + row * ncols], vmin=0, vmax=1)
                ax[col].set_title(pascal_ids[col + row * ncols])
                ax[col].axis('off')
        fig.savefig(os.path.join(save_segmentation_true_dir, f'seg_true_iteration_{i}.jpg'))
        plt.close(fig)

        # fig, ax = plt.subplots(nrows, ncols)
        # fig.set_size_inches((8, 8), forward=False)
        # fig.suptitle(layer, fontsize=20)
        # for row in range(nrows):
        #     for col in range(ncols):
        #         ax[row, col].imshow(y_pred[i, :, :, col + row * ncols], vmin=0, vmax=1, cmap='Greys')
        #         ax[row, col].axis('off')
        # save_path = os.path.join(save_layers_dir, layer)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # plt.savefig(
        #     os.path.join(save_path, '{}.jpg'.format(i)))
        # plt.close(fig)
        # plt.show()
        # plt.subplot(121)
        # plt.imshow(y_pred[i], vmin=0, vmax=20)
        # plt.subplot(122)
        # plt.imshow(y[i], vmin=0, vmax=20)
        # plt.show()
        # print(y_pred.shape)


if __name__ == '__main__':
    model = build_model(90, 90, 27)
    model.load_weights(
        '/home/pseweryn/Projects/multidimensional_lstm/repository/models/one_class_only_without_permute/weights.01-0.4970.hdf5'
    )
    layers = ['lambda_16']

    for layer in layers:
        model = get_model_with_layer(
            '/home/pseweryn/Projects/multidimensional_lstm/repository/models/one_class_only/weights.03-0.4766.hdf5',
            layer,
        )
    validate(model, batch_size=1, layer=layer)
