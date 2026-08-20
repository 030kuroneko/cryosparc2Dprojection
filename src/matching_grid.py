from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class MatchingGrid:
    class_average: np.ndarray
    volume: np.ndarray
    pixel_size: float


def prepare_matching_grid(
    class_average,
    volume,
    *,
    class_pixel_size,
    volume_pixel_size,
    max_size=128,
):
    """Put a class average and map on one bounded, physical matching grid."""
    class_average = np.asarray(class_average, dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)
    class_pixel_size = float(class_pixel_size)
    volume_pixel_size = float(volume_pixel_size)
    if class_pixel_size <= 0 or volume_pixel_size <= 0:
        raise ValueError("pixel sizes must be positive")
    if class_average.ndim != 2 or class_average.shape[0] != class_average.shape[1]:
        raise ValueError("class average must be a square 2D image")
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("volume must be a cubic 3D array")
    if max_size < 1:
        raise ValueError("max size must be positive")

    class_size = class_average.shape[0]
    downsample = max(1.0, class_size / int(max_size))
    output_pixel_size = class_pixel_size * downsample
    output_size = max(1, int(round(class_size / downsample)))

    prepared_class = _resample(class_average, class_pixel_size / output_pixel_size)
    prepared_volume = _resample(volume, volume_pixel_size / output_pixel_size)
    prepared_class = _center_fit(prepared_class, (output_size, output_size))
    prepared_volume = _center_fit(
        prepared_volume, (output_size, output_size, output_size)
    )
    return MatchingGrid(
        class_average=prepared_class.astype(np.float32, copy=False),
        volume=prepared_volume.astype(np.float32, copy=False),
        pixel_size=output_pixel_size,
    )


def _resample(array, factor):
    if np.isclose(factor, 1.0):
        return array.copy()
    return zoom(array, zoom=factor, order=1, mode="constant", cval=0.0, prefilter=False)


def _center_fit(array, shape):
    result = np.zeros(shape, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(array.shape, shape, strict=True):
        copied_size = min(source_size, target_size)
        source_start = (source_size - copied_size) // 2
        target_start = (target_size - copied_size) // 2
        source_slices.append(slice(source_start, source_start + copied_size))
        target_slices.append(slice(target_start, target_start + copied_size))
    result[tuple(target_slices)] = array[tuple(source_slices)]
    return result
