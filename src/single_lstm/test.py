import os

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

from single_lstm.models import build_model, get_model_with_layer
from utils.images import Dataset
from utils.pascal_voc import get_classes, resolve_class_names

plt.rcParams.update({'font.size': 6})

validation_data_dir = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'
save_directory = 'output/single_lstm'
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
    dataset = Dataset(
        validation_data_dir, 'val', chosen_classes=resolve_class_names(['person']), image_shape=(270, 270)
    )
    X, y = dataset.generate_data(100)
    print(X.shape)

    y_pred = model.predict([X[0], X[1], X[2], X[3]], batch_size=batch_size, verbose=1)

    print(y_pred.shape)

    for i in range(y_pred.shape[0]):
        fig, ax = plt.subplots(nrows, ncols)
        fig.set_size_inches((8, 2), forward=False)
        for row in range(nrows):
            for col in range(ncols):
                ax[col].imshow(y[i, :, :, col + row * ncols], vmin=0, vmax=1)
                ax[col].set_title(get_classes()[col + row * ncols])
                ax[col].axis('off')
        fig.savefig(os.path.join(save_segmentation_true_dir, f'seg_true_iteration_{i}.jpg'))
        plt.close(fig)


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
