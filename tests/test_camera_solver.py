import numpy as np
from scipy.ndimage import rotate, shift
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.camera import solve_class_camera


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
