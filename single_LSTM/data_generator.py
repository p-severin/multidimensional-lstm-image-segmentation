import os
from functools import partial
from typing import ClassVar

import keras
import numpy as np
import pandas as pd
from PIL import Image
from skimage.util import view_as_windows
from sklearn.externals._pilutil import imresize
from tensorflow.python.keras.utils import to_categorical


class DataGenerator(keras.utils.Sequence):
    extensions: ClassVar[dict] = dict(image='.jpg', segmentation='.png', annotations='.txt')

    def __init__(self, directory_images, subset, batch_size=16, dim=(90, 90), n_channels=3, n_classes=21, shuffle=True):
        self.dim = dim
        self.subset = subset
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.main_directory = partial(os.path.join, directory_images)
        self.directories = dict(
            image=self.main_directory('JPEGImages'),
            segmentation=self.main_directory('SegmentationClass'),
            annotations=self.main_directory('ImageSets/Segmentation'),
        )
        self.list_of_paths = self.get_image_numbers(subset)
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_of_paths) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_of_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_of_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_paths):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X_v = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X_h = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X_vh = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(temp_paths):
            temp_image = self.open_image(ID, 'image')
            temp_image = imresize(temp_image, (self.dim[0] * 3, self.dim[1] * 3))
            segmentation = self.open_image(ID, 'segmentation')
            segmentation = imresize(segmentation, self.dim, interp='nearest')
            segmentation[segmentation == 255] = 0

            patches_rgb, patches_segmentation = self.extract_patches((temp_image, segmentation))
            patches_rgb = np.multiply(patches_rgb, 1.0 / 255)
            patches_segmentation = self.__one_hot_encode_y(patches_segmentation)
            image_v = np.flip(patches_rgb, axis=0)
            image_h = np.flip(patches_rgb, axis=1)
            image_vh = np.flip(image_v, axis=1)
            X[i,] = patches_rgb
            X_v[i,] = image_v
            X_h[i,] = image_h
            X_vh[i,] = image_vh
            y[i,] = patches_segmentation

        return [X, X_v, X_h, X_vh], y

    def open_image(self, file_name, image_type):
        image_path = os.path.join(self.directories[image_type], file_name + self.extensions[image_type])
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        return image

    def get_image_numbers(self, subset):

        if subset not in ['train', 'trainval', 'val']:
            raise Exception('No such data subset exists.')

        train_file = os.path.join(self.directories['annotations'], subset + self.extensions['annotations'])
        csv_data = pd.read_csv(train_file, header=None)
        return csv_data.values.reshape(-1)

    def __one_hot_encode_y(self, segmentation_images):
        one_hot_encoded = to_categorical(segmentation_images, num_classes=self.n_classes)
        return one_hot_encoded

    def extract_patches(self, pair_of_images):
        rgb_image = pair_of_images[0]
        segmentation = pair_of_images[1]
        patch_shape = (3, 3, 3)
        segmentation_shape = (1, 1)
        step = 3
        segmentation_step = 1
        patches_rgb = view_as_windows(rgb_image, patch_shape, step=step)
        patches_rgb = np.reshape(patches_rgb, (patches_rgb.shape[0], patches_rgb.shape[1], np.prod(patch_shape)))
        patches_segmentation = view_as_windows(segmentation, segmentation_shape, step=segmentation_step)
        patches_segmentation = np.reshape(
            patches_segmentation,
            (patches_segmentation.shape[0], patches_segmentation.shape[1], np.prod(segmentation_shape)),
        )
        return patches_rgb, patches_segmentation


if __name__ == '__main__':
    directory_voc_dataset = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'
    datagen = DataGenerator(directory_voc_dataset, 'train', n_channels=27, dim=(90, 90))
    print(datagen.__len__())
    X, y = datagen.__getitem__(1)
    print(X.shape)
    print(y.shape)
