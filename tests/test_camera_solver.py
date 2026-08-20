import numpy as np
from scipy.ndimage import rotate, shift
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.camera import (
    fold_camera_rotations,
    solve_class_camera,
    solve_class_camera_from_particle_poses,
)


def test_symmetry_equivalent_complete_cameras_are_folded_before_averaging():
    cameras = Rotation.from_euler(
        "z", [[0], [120], [240]], degrees=True
    ).as_matrix()

    folded = fold_camera_rotations(cameras, "C3")

    assert np.allclose(folded, np.eye(3), atol=1e-6)


def test_class_camera_solver_reproduces_an_identity_camera():
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[1, 2, 3] = 1.0
    volume[4, 1, 5] = 2.0
    volume[5, 5, 1] = 3.0
    class_average = volume.sum(axis=0)

    result = solve_class_camera(
        class_average,
        volume,
        initial_rotation=np.eye(3),
        symmetry="C1",
    )

    assert np.allclose(result.rotation_matrix, np.eye(3), atol=1e-6)
    assert np.allclose(result.matched_projection, class_average, atol=1e-6)
    assert np.allclose(result.projection_shift_pixels, [0.0, 0.0], atol=1e-6)
    assert np.isclose(result.match_score, 1.0)


def test_class_camera_solver_preserves_in_plane_rotation():
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    volume[1, 1, 2] = 1.0
    volume[4, 2, 5] = 2.0
    volume[5, 5, 1] = 4.0
    untilted_projection = volume.sum(axis=0)
    class_average = np.rot90(untilted_projection)
    # Image rows increase downward, so a pixel-space counter-clockwise turn is
    # a negative Cartesian rotation about the viewing axis.
    camera_rotation = Rotation.from_euler("z", -90, degrees=True).as_matrix()

    result = solve_class_camera(
        class_average,
        volume,
        initial_rotation=camera_rotation,
        symmetry="C1",
    )

    assert np.allclose(result.rotation_matrix, camera_rotation, atol=1e-6)
    assert np.allclose(result.matched_projection, class_average, atol=1e-6)
    assert np.isclose(result.match_score, 1.0)
    assert np.allclose(result.quaternion_xyzw, [0.0, 0.0, -2**-0.5, 2**-0.5])
    assert np.allclose(result.view_direction, [0.0, 0.0, 1.0])
    assert np.isclose(result.in_plane_rotation_degrees, -90.0)


def test_class_camera_solver_reports_projection_shift_separately_from_rotation():
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[2, 2, 3] = 1.0
    volume[5, 4, 6] = 3.0
    projection = volume.sum(axis=0)
    class_average = shift(
        projection,
        shift=(1, -2),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    result = solve_class_camera(
        class_average,
        volume,
        initial_rotation=np.eye(3),
        symmetry="C1",
    )

    assert np.allclose(result.rotation_matrix, np.eye(3), atol=1e-6)
    assert np.allclose(result.projection_shift_pixels, [-2.0, 1.0], atol=1e-6)
    assert np.allclose(result.matched_projection, class_average, atol=1e-6)
    assert np.isclose(result.match_score, 1.0)


def test_class_camera_solver_refines_in_plane_rotation_around_the_pose_seed():
    volume = np.zeros((21, 21, 21), dtype=np.float32)
    volume[4:8, 3:6, 8:10] = 1.0
    volume[12:17, 13:18, 2:5] = 3.0
    expected_rotation = Rotation.from_euler("z", -10, degrees=True).as_matrix()
    class_average = rotate(
        volume.sum(axis=0),
        angle=10,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    result = solve_class_camera(
        class_average,
        volume,
        initial_rotation=np.eye(3),
        symmetry="C1",
        local_angular_range_degrees=15,
        local_angular_step_degrees=5,
    )

    assert np.allclose(result.rotation_matrix, expected_rotation, atol=1e-6)
    assert result.match_score > 0.99


def test_class_camera_solver_refines_view_direction_around_the_pose_seed():
    volume = np.zeros((21, 21, 21), dtype=np.float32)
    volume[3:7, 2:5, 8:11] = 1.0
    volume[13:18, 14:18, 3:6] = 4.0
    class_average = volume.sum(axis=0)
    tilted_seed = Rotation.from_euler("x", 5, degrees=True).as_matrix()

    result = solve_class_camera(
        class_average,
        volume,
        initial_rotation=tilted_seed,
        symmetry="C1",
        local_angular_range_degrees=5,
        local_angular_step_degrees=5,
    )

    assert np.allclose(result.rotation_matrix, np.eye(3), atol=1e-6)
    assert result.match_score > 0.99
    assert result.second_best_score < result.match_score
    assert result.score_margin > 0
    assert result.match_confidence == "high"
    assert result.search_evaluation_count <= 20


def test_class_camera_solver_combines_3d_poses_with_2d_in_plane_poses():
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[1, 2, 3] = 1.0
    volume[5, 1, 6] = 2.0
    volume[6, 6, 2] = 4.0
    class_average = np.rot90(volume.sum(axis=0))

    result = solve_class_camera_from_particle_poses(
        class_average,
        volume,
        refinement_poses=np.array([[0.0, 0.0, 0.0]]),
        alignment_2d_poses=np.array([np.pi / 2]),
        symmetry="C1",
        local_angular_range_degrees=0,
    )

    expected = Rotation.from_euler("z", -90, degrees=True).as_matrix()
    assert np.allclose(result.rotation_matrix, expected, atol=1e-6)
    assert np.allclose(result.matched_projection, class_average, atol=1e-6)
