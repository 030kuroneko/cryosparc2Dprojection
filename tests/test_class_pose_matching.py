import numpy as np
import pytest

from cryosparc_2d_projection.class_poses import (
    analyze_class_orientations,
    match_class_poses,
)


def test_only_overlapping_particle_uids_contribute_to_each_2d_class():
    select_2d = np.array(
        [
            (101, 0),
            (102, 0),
            (103, 1),
            (999, 2),
        ],
        dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
    )
    refinement = np.array(
        [
            (103, [0.0, 0.0, 0.3]),
            (101, [0.1, 0.0, 0.0]),
            (777, [0.0, 0.2, 0.0]),
        ],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )

    matched = match_class_poses(select_2d, refinement)

    assert set(matched) == {0, 1}
    assert matched[0].particle_uids.tolist() == [101]
    assert np.allclose(matched[0].poses, [[0.1, 0.0, 0.0]])
    assert matched[1].particle_uids.tolist() == [103]
    assert np.allclose(matched[1].poses, [[0.0, 0.0, 0.3]])


def test_class_orientation_uses_cryosparc_rodrigues_pose_convention():
    select_2d = np.array(
        [(101, 0), (102, 0)],
        dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
    )
    refinement = np.array(
        [
            (101, [0.0, 0.0, 0.0]),
            (102, [0.0, np.pi / 2, 0.0]),
        ],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )

    orientations = analyze_class_orientations(select_2d, refinement)

    assert orientations[0].particle_count == 2
    assert np.allclose(
        orientations[0].view_direction,
        [np.sqrt(0.5), 0.0, np.sqrt(0.5)],
    )
    assert np.isclose(orientations[0].angular_spread_degrees, 45.0)


@pytest.mark.parametrize("symmetry", ["C0", "D0", "C-2", "D1.5", "I1", "I2", "C01", "", "Cn"])
def test_orientation_analysis_rejects_unsupported_symmetry(symmetry):
    select_2d = np.array(
        [(101, 0)],
        dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
    )
    refinement = np.array(
        [(101, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )

    with pytest.raises(ValueError, match="Supported symmetry:"):
        analyze_class_orientations(select_2d, refinement, symmetry=symmetry)


def test_symmetry_equivalent_directions_are_folded_before_averaging():
    select_2d = np.array(
        [(101, 0), (102, 0)],
        dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
    )
    refinement = np.array(
        [
            (101, [0.0, np.pi / 2, 0.0]),
            (102, [0.0, -np.pi / 2, 0.0]),
        ],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )

    orientations = analyze_class_orientations(
        select_2d,
        refinement,
        symmetry="I",
    )

    assert np.allclose(orientations[0].view_direction, [1.0, 0.0, 0.0])
    assert np.isclose(orientations[0].angular_spread_degrees, 0.0)


def test_orientation_analysis_rejects_datasets_without_overlapping_uids():
    select_2d = np.array(
        [(101, 0)],
        dtype=[("uid", "u8"), ("alignments2D/class", "i4")],
    )
    refinement = np.array(
        [(202, [0.0, 0.0, 0.0])],
        dtype=[("uid", "u8"), ("alignments3D/pose", "f8", (3,))],
    )

    with pytest.raises(ValueError, match="No overlapping particle UIDs"):
        analyze_class_orientations(select_2d, refinement)


@pytest.mark.parametrize("symmetry,mate", [
    ("C2", np.diag([-1., -1., 1.])),
    ("D7", np.diag([-1., 1., -1.])),
    ("T", np.array([[-1., 0., 0.], [0., 1/3, 2*np.sqrt(2)/3], [0., 2*np.sqrt(2)/3, -1/3]])),
    ("O", np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])),
])
def test_new_point_groups_fold_equivalent_particle_directions(symmetry, mate):
    from scipy.spatial.transform import Rotation
    reference = Rotation.from_euler("xyz", [17, 31, 43], degrees=True).as_matrix()
    cameras = np.array([reference, reference @ mate])
    poses = Rotation.from_matrix(cameras.transpose(0, 2, 1)).as_rotvec()
    selected = {"uid": np.array([1, 2]), "alignments2D/class": np.array([0, 0])}
    refined = {"uid": np.array([1, 2]), "alignments3D/pose": poses}
    result = analyze_class_orientations(selected, refined, symmetry=symmetry)[0]
    np.testing.assert_allclose(result.view_direction, reference[2], atol=1e-12)
    assert result.angular_spread_degrees < 1e-5
