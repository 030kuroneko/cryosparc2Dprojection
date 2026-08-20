from dataclasses import dataclass, replace

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
    second_best_score: float | None = None
    score_margin: float | None = None
    match_confidence: str = "low"
    search_evaluation_count: int = 0


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
        particle_cameras = fold_camera_rotations(particle_cameras, symmetry)
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


def fold_camera_rotations(camera_matrices, symmetry):
    """Fold complete camera rotations into one symmetry-equivalent neighborhood."""
    camera_matrices = np.asarray(camera_matrices, dtype=float)
    if camera_matrices.ndim != 3 or camera_matrices.shape[1:] != (3, 3):
        raise ValueError("camera rotations must have shape (N, 3, 3)")
    if len(camera_matrices) == 0:
        raise ValueError("at least one camera rotation is required")

    group_name = symmetry.strip().upper()
    if group_name == "C1":
        return camera_matrices.copy()
    if group_name in {"I1", "I2"}:
        group_name = "I"
    symmetry_matrices = Rotation.create_group(group_name).as_matrix()
    folded = camera_matrices.copy()
    reference = camera_matrices[0]

    for _ in range(2):
        for index, camera in enumerate(camera_matrices):
            equivalents = camera @ symmetry_matrices
            distances = Rotation.from_matrix(
                equivalents @ reference.T
            ).magnitude()
            folded[index] = equivalents[np.argmin(distances)]
        reference = Rotation.from_matrix(folded).mean().as_matrix()
    return folded


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

    candidates = {}

    def evaluate(rotation_matrix):
        key = tuple(np.round(rotation_matrix, decimals=10).ravel())
        if key in candidates:
            return candidates[key]
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
        result = ClassCameraResult(
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
        candidates[key] = result
        return result

    evaluate(initial_rotation)
    beam = [(np.zeros(3), evaluate(initial_rotation))]
    for axis in (2, 0, 1):
        expanded = []
        for angles, _ in beam:
            for delta in deltas:
                candidate_angles = angles.copy()
                candidate_angles[axis] = delta
                perturbation = Rotation.from_euler(
                    "xyz", candidate_angles, degrees=True
                ).as_matrix()
                expanded.append(
                    (candidate_angles, evaluate(perturbation @ initial_rotation))
                )
        expanded.sort(key=lambda item: item[1].match_score, reverse=True)
        beam = expanded[:2]

    ranked_candidates = sorted(
        candidates.values(), key=lambda candidate: candidate.match_score, reverse=True
    )
    best = ranked_candidates[0]
    second_best_score = (
        ranked_candidates[1].match_score if len(ranked_candidates) > 1 else None
    )
    score_margin = (
        best.match_score - second_best_score
        if second_best_score is not None
        else None
    )
    match_confidence = (
        "high"
        if best.match_score >= 0.5
        and (score_margin is None or score_margin >= 0.01)
        else "low"
    )
    return replace(
        best,
        second_best_score=second_best_score,
        score_margin=score_margin,
        match_confidence=match_confidence,
        search_evaluation_count=len(ranked_candidates),
    )


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
