import os

import numpy as np
from PIL import Image
from skimage.util import view_as_windows
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

SUBSET_DIR_MAP = {
    'train': ('train_images', 'train_labels'),
    'val': ('valid_images', 'valid_labels'),
}


def imresize(arr, size, interp='bilinear'):
    resample_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
    }
    resample = resample_map.get(interp, Image.BILINEAR)
    im = Image.fromarray(arr)
    im = im.resize((size[1], size[0]), resample=resample)
    return np.array(im)


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        data_dir: str,
        subset: str,
        chosen_classes: list[int],
        batch_size: int = 16,
        dim: tuple[int, int] = (90, 90),
        shuffle: bool = True,
    ):
        if subset not in SUBSET_DIR_MAP:
            raise ValueError(f"subset must be one of {list(SUBSET_DIR_MAP.keys())}, got '{subset}'")

        self.dim = dim
        self.batch_size = batch_size
        self.chosen_classes = chosen_classes
        self.n_classes = len(chosen_classes) + 1
        self.n_channels = 27
        self.shuffle = shuffle

        images_dir_name, labels_dir_name = SUBSET_DIR_MAP[subset]
        self.images_dir = os.path.join(data_dir, images_dir_name)
        self.labels_dir = os.path.join(data_dir, labels_dir_name)

        self.file_stems = self._discover_and_filter_files()
        self.on_epoch_end()

    def _discover_and_filter_files(self) -> list[str]:
        all_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith('.jpg'))
        chosen_set = set(self.chosen_classes)
        filtered = []
        for stem in all_stems:
            mask = self._load_mask_raw(stem)
            if chosen_set.intersection(np.unique(mask)):
                filtered.append(stem)
        return filtered

    def _load_mask_raw(self, stem: str) -> np.ndarray:
        path = os.path.join(self.labels_dir, stem + '.png')
        return np.array(Image.open(path), dtype=np.uint8)

    def __len__(self):
        return len(self.file_stems) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_stems = [self.file_stems[k] for k in indexes]
        return self._generate_batch(batch_stems)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_stems))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_image(self, stem: str) -> np.ndarray:
        path = os.path.join(self.images_dir, stem + '.jpg')
        return np.array(Image.open(path))

    def _remap_classes(self, mask: np.ndarray) -> np.ndarray:
        mask[mask == 255] = 0
        remapped = np.zeros_like(mask)
        for new_idx, original_class in enumerate(self.chosen_classes, start=1):
            remapped[mask == original_class] = new_idx
        return remapped

    def _generate_batch(self, batch_stems: list[str]):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X_v = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X_h = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X_vh = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)

        for i, stem in enumerate(batch_stems):
            image = self._load_image(stem)
            image = imresize(image, (self.dim[0] * 3, self.dim[1] * 3))

            mask = self._load_mask_raw(stem)
            mask = imresize(mask, self.dim, interp='nearest')
            mask = self._remap_classes(mask)

            patches_rgb, patches_seg = self._extract_patches(image, mask)
            patches_rgb = patches_rgb.astype(np.float32) / 255.0

            patches_seg = to_categorical(patches_seg, num_classes=self.n_classes)

            X[i] = patches_rgb
            X_v[i] = np.flip(patches_rgb, axis=0)
            X_h[i] = np.flip(patches_rgb, axis=1)
            X_vh[i] = np.flip(np.flip(patches_rgb, axis=0), axis=1)
            y[i] = patches_seg

        return [X, X_v, X_h, X_vh], y

    @staticmethod
    def _extract_patches(image: np.ndarray, mask: np.ndarray):
        patches_rgb = view_as_windows(image, (3, 3, 3), step=3)
        patches_rgb = patches_rgb.reshape(patches_rgb.shape[0], patches_rgb.shape[1], 27)

        patches_seg = view_as_windows(mask, (1, 1), step=1)
        patches_seg = patches_seg.reshape(patches_seg.shape[0], patches_seg.shape[1], 1)

        return patches_rgb, patches_seg
