from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import shift as shift_image
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class NativeMatchedProjection:
    """Native-grid projection regenerated after a camera was selected."""

    projection: np.ndarray
    matched_projection: np.ndarray
    projection_shift_pixels: np.ndarray
    projection_pixel_size_A: float


def project_native_matched_projection(
    class_average,
    volume,
    rotation_matrix,
    *,
    class_pixel_size,
    volume_pixel_size,
):
    """Regenerate a Matched Projection at the native class grid.

    ``rotation_matrix`` is treated as fixed: the only optimization performed
    here is the two-dimensional translation that aligns the projection to the
    Class Average.  Camera selection and bounded Search Projection scoring are
    intentionally outside this seam.
    """
    from cryosparc_2d_projection.matching_grid import prepare_native_matching_grid

    grid = prepare_native_matching_grid(
        class_average,
        volume,
        class_pixel_size=class_pixel_size,
        volume_pixel_size=volume_pixel_size,
    )
    projection = project_volume_at_rotation(grid.volume, rotation_matrix)
    shift_xy = find_projection_shift(grid.class_average, projection)
    matched_projection = shift_image(
        projection,
        shift=(shift_xy[1], shift_xy[0]),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return NativeMatchedProjection(
        projection=projection.astype(np.float32, copy=False),
        matched_projection=matched_projection.astype(np.float32, copy=False),
        projection_shift_pixels=shift_xy,
        projection_pixel_size_A=grid.pixel_size,
    )


def project_volume_at_rotation(volume, rotation_matrix):
    """Project a volume after applying a complete Cartesian camera rotation."""
    return rotate_volume_at_rotation(volume, rotation_matrix).sum(axis=0)


def rotate_volume_at_rotation(volume, rotation_matrix):
    """Rotate a volume into a complete Cartesian camera orientation."""
    volume = np.asarray(volume)
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    reverse_axes = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    output_to_input = reverse_axes @ rotation_matrix.T @ reverse_axes
    center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    return affine_transform(
        volume,
        matrix=output_to_input,
        offset=center - output_to_input @ center,
        output_shape=volume.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def project_volume(volume, view_direction):
    """Project a 3D MRC-style array from the requested viewing direction."""
    volume = np.asarray(volume)
    direction = np.asarray(view_direction, dtype=float)
    direction /= np.linalg.norm(direction)

    if np.allclose(direction, [0.0, 0.0, 1.0]):
        return volume.sum(axis=0)

    reference = (
        np.array([1.0, 0.0, 0.0])
        if abs(direction[1]) > 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    image_x = np.cross(reference, direction)
    image_x /= np.linalg.norm(image_x)
    image_y = np.cross(direction, image_x)

    matrix = np.array(
        [
            [direction[2], image_y[2], image_x[2]],
            [direction[1], image_y[1], image_x[1]],
            [direction[0], image_y[0], image_x[0]],
        ]
    )
    center = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    rotated = affine_transform(
        volume,
        matrix=matrix,
        offset=center - matrix @ center,
        output_shape=volume.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return rotated.sum(axis=0)


def find_projection_shift(target, source):
    """Return the XY pixel shift that aligns ``source`` to ``target``."""
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)
    if target.ndim != 2 or source.ndim != 2 or target.shape != source.shape:
        raise ValueError("target and source projections must have the same 2D shape")
    correlation = fftconvolve(target, source[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    return np.array(
        [
            peak_x - (source.shape[1] - 1),
            peak_y - (source.shape[0] - 1),
        ],
        dtype=float,
    )
