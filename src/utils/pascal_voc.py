import json
import os

import numpy as np

_CLASSES: list[str] | None = None
_COLORS: list[tuple[int, int, int]] | None = None
_RGB_LUT: np.ndarray | None = None


def load(data_dir: str = 'data') -> None:
    global _CLASSES, _COLORS, _RGB_LUT

    with open(os.path.join(data_dir, 'classes_pascal.json')) as f:
        classes: list[str] = json.load(f)

    with open(os.path.join(data_dir, 'segmentation_colors.json')) as f:
        raw_colors = json.load(f)
    colors: list[tuple[int, int, int]] = [tuple(c) for c in raw_colors]

    if len(classes) != len(colors):
        raise ValueError(
            f'classes_pascal.json has {len(classes)} entries but segmentation_colors.json has {len(colors)}'
        )

    _CLASSES = classes
    _COLORS = colors
    _RGB_LUT = _build_rgb_lut(colors)


def _ensure_loaded() -> None:
    if _CLASSES is None:
        load()


def _build_rgb_lut(colors: list[tuple[int, int, int]]) -> np.ndarray:
    lut = np.zeros(256 * 256 * 256, dtype=np.uint8)
    for idx, (r, g, b) in enumerate(colors):
        lut[r * 65536 + g * 256 + b] = idx
    return lut


def get_classes() -> list[str]:
    _ensure_loaded()
    assert _CLASSES is not None
    return _CLASSES


def get_colors() -> list[tuple[int, int, int]]:
    _ensure_loaded()
    assert _COLORS is not None
    return _COLORS


def class_name_to_index(name: str) -> int:
    _ensure_loaded()
    assert _CLASSES is not None
    try:
        return _CLASSES.index(name)
    except ValueError:
        raise ValueError(f"Unknown class name: '{name}'. Available: {_CLASSES}") from None


def resolve_class_names(names: list[str]) -> list[int]:
    return [class_name_to_index(name) for name in names]


def rgb_mask_to_indices(mask_rgb: np.ndarray) -> np.ndarray:
    _ensure_loaded()
    assert _RGB_LUT is not None
    flat = (
        mask_rgb[:, :, 0].astype(np.int32) * 65536
        + mask_rgb[:, :, 1].astype(np.int32) * 256
        + mask_rgb[:, :, 2].astype(np.int32)
    )
    return _RGB_LUT[flat]


def index_mask_to_rgb(mask_indices: np.ndarray) -> np.ndarray:
    _ensure_loaded()
    assert _COLORS is not None
    color_array = np.array(_COLORS, dtype=np.uint8)
    safe = np.clip(mask_indices, 0, len(color_array) - 1)
    return color_array[safe]
