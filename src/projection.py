import numpy as np
from scipy.ndimage import affine_transform


def project_volume_at_rotation(volume, rotation_matrix):
    """Project a volume after applying a complete Cartesian camera rotation."""
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
    rotated = affine_transform(
        volume,
        matrix=output_to_input,
        offset=center - output_to_input @ center,
        output_shape=volume.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return rotated.sum(axis=0)


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
