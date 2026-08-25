import numpy as np
import pytest

from cryosparc_2d_projection.axis_projection import (
    AxisReferenceProjection,
    project_axis_reference,
)


def test_axis_reference_uses_matching_map_and_exposes_raw_and_cryosparc_display_arrays():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[1, 2, 3] = 1.0
    volume[3, 4, 0] = 2.0

    result = project_axis_reference(
        volume,
        "2fold",
        pixel_size_A=1.5,
    )

    assert isinstance(result, AxisReferenceProjection)
    assert result.source_map == "matching"
    assert result.pixel_size_A == 1.5
    assert result.projection.shape == (5, 5)
    assert np.array_equal(result.display_projection, np.flipud(result.projection))
    assert np.allclose(result.rotation_matrix[2], [0.0, 1.0, 0.0])


def test_axis_reference_always_returns_display_orientation_without_mutating_raw_projection():
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    result = project_axis_reference(volume, "5fold")

    assert result.projection.shape == (3, 3)
    assert np.array_equal(
        result.display_projection,
        np.flipud(result.projection),
    )


def test_axis_reference_display_orientation_is_not_caller_selectable():
    volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    with pytest.raises(TypeError):
        project_axis_reference(volume, "2fold", display_orientation=False)
