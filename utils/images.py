import os
from functools import partial
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.util import view_as_windows
from tensorflow.keras.utils import to_categorical

from single_lstm.data_generator import imresize


class Dataset:
    extensions: ClassVar[dict] = dict(image='.jpg', segmentation='.png', annotations='.txt')

    def __init__(self, directory_images: str, subset: str, chosen_classes: list, image_shape=(270, 270)):
        self.main_directory = partial(os.path.join, directory_images)
        self.directories = dict(
            image=self.main_directory('JPEGImages'),
            segmentation=self.main_directory('SegmentationClass'),
            annotations=self.main_directory('ImageSets/Segmentation'),
        )
        self.subset = subset
        self.chosen_classes = chosen_classes
        self.image_shape = image_shape
        self.segmentation_shape = (self.image_shape[0] // 3, self.image_shape[1] // 3)
        self.files = self.__get_image_numbers()
        self.original_images: list | np.ndarray = []
        self.X: list | np.ndarray = []
        self.X_vertical: list | np.ndarray = []
        self.X_horizontal: list | np.ndarray = []
        self.X_both_transformations: list | np.ndarray = []
        self.y: list | np.ndarray = []

    def __get_image_numbers(self):
        if self.subset not in ['train', 'trainval', 'val']:
            raise Exception('No such data subset exists.')

        train_file = os.path.join(self.directories['annotations'], self.subset + self.extensions['annotations'])
        csv_data = pd.read_csv(train_file, header=None)
        return csv_data.values.reshape(-1)

    def __open_image(self, file_name, image_type):
        image_path = os.path.join(self.directories[image_type], file_name + self.extensions[image_type])
        image = Image.open(image_path)
        image = np.array(image)
        return image

    def generate_images(self, how_many_images: int):
        if how_many_images == -1:
            number_of_elements = len(self.files)
        else:
            number_of_elements = how_many_images

        for file in self.files[:number_of_elements]:
            image = self.__open_image(file, 'image')
            segmentation = self.__open_image(file, 'segmentation')
            self.X.append(image)
            self.y.append(segmentation)

    def resize_images(self):
        self.X = [imresize(image, self.image_shape, interp='bilinear') for image in self.X]
        self.y = [imresize(image, self.segmentation_shape, interp='nearest') for image in self.y]

    def remove_classes_not_used(self):
        assert isinstance(self.y, np.ndarray)
        classes_to_be_removed = list(np.unique(self.y))
        classes_to_be_removed = [x for x in classes_to_be_removed if x not in self.chosen_classes]
        print(classes_to_be_removed)
        for not_chosen in classes_to_be_removed:
            self.y[self.y == not_chosen] = 0

    def transform_data_to_numpy_arrays(self):
        self.X = np.array(self.X)
        self.X_vertical = np.array(self.X_vertical)
        self.X_horizontal = np.array(self.X_horizontal)
        self.X_both_transformations = np.array(self.X_both_transformations)
        self.y = np.array(self.y)

    def prepare_input_data_X(self):
        self.X = np.multiply(self.X, 1.0 / 255)

    def create_flipped_windows(self):
        self.X_vertical = np.flip(self.X, axis=1)
        self.X_horizontal = np.flip(self.X, axis=2)
        self.X_both_transformations = np.flip(self.X_vertical, axis=2)

    def divide_image_into_patches(self, image, patch_shape, step):
        patches = view_as_windows(image, patch_shape, step=step)
        patches = np.reshape(patches, (patches.shape[0], patches.shape[1], np.prod(patch_shape)))
        return patches

    def leave_images_of_one_class(self):
        newX = []
        newY = []
        for image, segmentation in zip(self.X, self.y, strict=False):
            for chosen_class in self.chosen_classes:
                if chosen_class in np.unique(segmentation):
                    newX.append(image)
                    newY.append(segmentation)
        self.X = np.array(newX)
        self.original_images = newX
        self.y = np.array(newY)
        for i, chosen in enumerate(self.chosen_classes):
            self.y[self.y == chosen] = i + 1

    def extract_patches_from_data(self):
        rgb_shape = (3, 3, 3)
        rgb_step = 3
        segmentation_shape = (1, 1)
        segmentation_step = 1

        self.X = [self.divide_image_into_patches(image, rgb_shape, rgb_step) for image in self.X]
        self.X_vertical = [self.divide_image_into_patches(image, rgb_shape, rgb_step) for image in self.X_vertical]
        self.X_horizontal = [self.divide_image_into_patches(image, rgb_shape, rgb_step) for image in self.X_horizontal]
        self.X_both_transformations = [
            self.divide_image_into_patches(image, rgb_shape, rgb_step) for image in self.X_both_transformations
        ]
        self.y = [self.divide_image_into_patches(image, segmentation_shape, segmentation_step) for image in self.y]

    def one_hot_encode_y(self):
        self.y = to_categorical(self.y, num_classes=len(self.chosen_classes) + 1)

    def generate_data(self, how_many_images=-1):
        self.generate_images(how_many_images=how_many_images)
        self.resize_images()
        self.transform_data_to_numpy_arrays()
        self.remove_classes_not_used()
        self.leave_images_of_one_class()
        self.prepare_input_data_X()
        self.create_flipped_windows()
        self.extract_patches_from_data()
        self.transform_data_to_numpy_arrays()
        self.one_hot_encode_y()
        return np.array([self.X, self.X_vertical, self.X_horizontal, self.X_both_transformations]), np.array(self.y)

    def plot_image_and_its_segmentation(self, image, segmentation_image):
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(segmentation_image)
        plt.show()


if __name__ == '__main__':
    data_dir = 'data/voc_2012_segmentation_data'
    # Note: Dataset class still uses old VOC2012 structure (JPEGImages/, SegmentationClass/, ImageSets/)
    # Use DataGenerator from single_lstm.data_generator for the new dataset format instead.
    print('Dataset class is deprecated. Use single_lstm.data_generator.DataGenerator instead.')
