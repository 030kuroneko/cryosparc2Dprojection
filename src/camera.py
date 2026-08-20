from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.ndimage import shift as shift_image
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation

from cryosparc_2d_projection.projection import project_volume_at_rotation


@dataclass(frozen=True)
class ClassCameraResult:
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    view_direction: np.ndarray
    in_plane_rotation_degrees: float
    matched_projection: np.ndarray
    projection_shift_pixels: np.ndarray
    match_score: float


def solve_class_camera_from_particle_poses(
    class_average,
    volume,
    *,
    refinement_poses,
    alignment_2d_poses,
    symmetry="C1",
    local_angular_range_degrees=15,
    local_angular_step_degrees=5,
):
    """Solve a class camera from overlapping CryoSPARC 2D and 3D poses."""
    refinement_poses = np.asarray(refinement_poses, dtype=float)
    alignment_2d_poses = np.asarray(alignment_2d_poses, dtype=float)
    if len(refinement_poses) != len(alignment_2d_poses) or len(refinement_poses) == 0:
        raise ValueError("2D and 3D poses must contain the same non-zero particle count")

    refinement_rotations = np.transpose(
        Rotation.from_rotvec(refinement_poses).as_matrix(), (0, 2, 1)
    )
    candidates = []
    for sign in (-1.0, 1.0):
        in_plane = Rotation.from_euler(
            "z", sign * alignment_2d_poses, degrees=False
        ).as_matrix()
        particle_cameras = in_plane @ refinement_rotations
        seed = Rotation.from_matrix(particle_cameras).mean().as_matrix()
        candidates.append(
            solve_class_camera(
                class_average,
                volume,
                initial_rotation=seed,
                symmetry=symmetry,
                local_angular_range_degrees=local_angular_range_degrees,
                local_angular_step_degrees=local_angular_step_degrees,
            )
        )
    return max(candidates, key=lambda candidate: candidate.match_score)


def solve_class_camera(
    class_average,
    volume,
    *,
    initial_rotation,
    symmetry="C1",
    local_angular_range_degrees=0,
    local_angular_step_degrees=1,
):
    """Solve one complete class camera from a pose-derived initial rotation."""
    initial_rotation = np.asarray(initial_rotation, dtype=float)
    if local_angular_range_degrees == 0:
        deltas = np.array([0.0])
    else:
        deltas = np.arange(
            -local_angular_range_degrees,
            local_angular_range_degrees + local_angular_step_degrees / 2,
            local_angular_step_degrees,
        )

    candidates = []
    for delta_x, delta_y, delta_z in product(deltas, repeat=3):
        perturbation = Rotation.from_euler(
            "xyz", [delta_x, delta_y, delta_z], degrees=True
        ).as_matrix()
        rotation_matrix = perturbation @ initial_rotation
        projection = project_volume_at_rotation(volume, rotation_matrix)
        shift_xy = _translation_to_match(class_average, projection)
        matched_projection = shift_image(
            projection,
            shift=(shift_xy[1], shift_xy[0]),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        candidates.append(
            ClassCameraResult(
                rotation_matrix=rotation_matrix,
                quaternion_xyzw=Rotation.from_matrix(rotation_matrix).as_quat(),
                view_direction=rotation_matrix[2].copy(),
                in_plane_rotation_degrees=float(
                    Rotation.from_matrix(rotation_matrix).as_euler("zyx", degrees=True)[0]
                ),
                matched_projection=matched_projection,
                projection_shift_pixels=shift_xy,
                match_score=_normalized_correlation(class_average, matched_projection),
            )
        )
    return max(candidates, key=lambda candidate: candidate.match_score)


def _translation_to_match(target, source):
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)
    correlation = fftconvolve(target, source[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    return np.array(
        [
            peak_x - (source.shape[1] - 1),
            peak_y - (source.shape[0] - 1),
        ],
        dtype=float,
    )


def _normalized_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.sum(left_centered * right_centered) / denominator, -1, 1))
