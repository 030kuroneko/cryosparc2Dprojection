import numpy as np

from cryosparc_2d_projection.symmetry import assign_symmetry_axis


def test_icosahedral_axes_are_named_only_within_the_threshold():
    two_fold = assign_symmetry_axis([0.0, 0.0, 1.0], "I", threshold_degrees=5)
    three_fold = assign_symmetry_axis([1.0, 1.0, 1.0], "I", threshold_degrees=5)
    five_fold = assign_symmetry_axis(
        [0.850650808, 0.0, 0.525731112], "I", threshold_degrees=5
    )
    ten_degrees_from_two_fold = assign_symmetry_axis(
        [np.sin(np.deg2rad(10)), 0.0, np.cos(np.deg2rad(10))],
        "I",
        threshold_degrees=5,
    )

    assert (two_fold.label, two_fold.nearest_order, two_fold.distance_degrees) == (
        "2-fold", 2, 0.0
    )
    assert three_fold.label == "3-fold"
    assert five_fold.label == "5-fold"
    assert ten_degrees_from_two_fold.label == "general"
    assert ten_degrees_from_two_fold.nearest_order == 2
    assert np.isclose(ten_degrees_from_two_fold.distance_degrees, 10.0)
