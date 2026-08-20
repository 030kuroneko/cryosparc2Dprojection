from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ClassPoses:
    particle_uids: np.ndarray
    poses: np.ndarray


@dataclass(frozen=True)
class ClassOrientation:
    particle_count: int
    view_direction: np.ndarray
    angular_spread_degrees: float


def match_class_poses(select_2d, refinement):
    """Group refinement poses by 2D class for particle UIDs present in both inputs."""
    refinement_rows = {
        int(uid): index for index, uid in enumerate(refinement["uid"])
    }
    grouped = {}

    for uid, class_id in zip(
        select_2d["uid"], select_2d["alignments2D/class"], strict=True
    ):
        row_index = refinement_rows.get(int(uid))
        if row_index is None:
            continue

        group = grouped.setdefault(int(class_id), {"uids": [], "poses": []})
        group["uids"].append(uid)
        group["poses"].append(refinement["alignments3D/pose"][row_index])

    return {
        class_id: ClassPoses(
            particle_uids=np.asarray(group["uids"]),
            poses=np.asarray(group["poses"]),
        )
        for class_id, group in grouped.items()
    }


def analyze_class_orientations(select_2d, refinement, *, symmetry="C1"):
    """Calculate a representative viewing direction for every matched 2D class."""
    matched = match_class_poses(select_2d, refinement)
    if not matched:
        raise ValueError("No overlapping particle UIDs between Select 2D and refinement")
    orientations = {}

    for class_id, class_poses in matched.items():
        view_directions = np.asarray(
            [_pose_to_view_direction(pose) for pose in class_poses.poses]
        )
        view_directions = _fold_symmetry_equivalents(view_directions, symmetry)
        mean_direction = view_directions.mean(axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        angular_distances = np.arccos(
            np.clip(view_directions @ mean_direction, -1.0, 1.0)
        )
        orientations[class_id] = ClassOrientation(
            particle_count=len(class_poses.particle_uids),
            view_direction=mean_direction,
            angular_spread_degrees=float(
                np.rad2deg(np.sqrt(np.mean(angular_distances**2)))
            ),
        )

    return orientations


def _fold_symmetry_equivalents(view_directions, symmetry):
    group_name = symmetry.strip().upper()
    if group_name == "C1":
        return view_directions
    if group_name in {"I1", "I2"}:
        group_name = "I"

    symmetry_matrices = Rotation.create_group(group_name).as_matrix()
    reference = view_directions[0]
    folded = view_directions.copy()

    for _ in range(2):
        for index, direction in enumerate(view_directions):
            equivalents = direction @ symmetry_matrices
            folded[index] = equivalents[np.argmax(equivalents @ reference)]
        reference = folded.mean(axis=0)
        reference /= np.linalg.norm(reference)

    return folded


def _pose_to_view_direction(pose):
    """Return the third row of CryoSPARC's Rodrigues rotation matrix."""
    pose = np.asarray(pose, dtype=float)
    theta = np.linalg.norm(pose)
    if theta < 1e-16:
        return np.array([0.0, 0.0, 1.0])

    axis = pose / theta
    sine = np.sin(theta)
    one_minus_cosine = 1.0 - np.cos(theta)
    x, y, z = axis
    return np.array(
        [
            sine * y + one_minus_cosine * x * z,
            -sine * x + one_minus_cosine * y * z,
            1.0 - one_minus_cosine * (x * x + y * y),
        ]
    )
