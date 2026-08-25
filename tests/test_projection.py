import numpy as np
from scipy.ndimage import shift
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.projection import (
    project_native_matched_projection,
    project_volume,
    project_volume_at_rotation,
)


def test_identity_view_projects_along_mrc_z_axis():
    volume = np.arange(27, dtype=float).reshape(3, 3, 3)

    projection = project_volume(volume, [0.0, 0.0, 1.0])

    assert np.allclose(projection, volume.sum(axis=0))


def test_x_axis_view_integrates_density_along_x():
    volume = np.broadcast_to(np.arange(3, dtype=float), (3, 3, 3))

    projection = project_volume(volume, [1.0, 0.0, 0.0])

    assert np.allclose(projection, np.full((3, 3), 3.0))


def test_native_matched_projection_reoptimizes_only_xy_shift_at_fixed_rotation():
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[2:4, 2:5, 3:5] = 1.0
    volume[6:9, 6:8, 7:9] = 2.0
    rotation = Rotation.from_euler("z", -12.0, degrees=True).as_matrix()
    native_projection = project_volume_at_rotation(volume, rotation)
    class_average = shift(
        native_projection,
        shift=(1.0, -2.0),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    result = project_native_matched_projection(
        class_average,
        volume,
        rotation,
        class_pixel_size=1.5,
        volume_pixel_size=1.5,
    )

    assert result.projection.shape == class_average.shape
    assert result.projection_pixel_size_A == 1.5
    assert np.allclose(result.projection_shift_pixels, [-2.0, 1.0])
    assert np.allclose(result.matched_projection, class_average, atol=1e-6)
