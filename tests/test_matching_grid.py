import numpy as np

from cryosparc_2d_projection.matching_grid import prepare_matching_grid


def test_matching_grid_resamples_class_and_volume_to_one_bounded_physical_grid():
    class_average = np.zeros((12, 12), dtype=np.float32)
    class_average[4:8, 4:8] = 1.0
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = 1.0

    prepared = prepare_matching_grid(
        class_average,
        volume,
        class_pixel_size=1.0,
        volume_pixel_size=2.0,
        max_size=6,
    )

    assert prepared.class_average.shape == (6, 6)
    assert prepared.volume.shape == (6, 6, 6)
    assert prepared.pixel_size == 2.0
    assert prepared.class_average.dtype == np.float32
    assert prepared.volume.dtype == np.float32


def test_matching_grid_rejects_invalid_pixel_sizes():
    with np.testing.assert_raises_regex(ValueError, "pixel sizes must be positive"):
        prepare_matching_grid(
            np.zeros((4, 4)),
            np.zeros((4, 4, 4)),
            class_pixel_size=0,
            volume_pixel_size=1,
        )
