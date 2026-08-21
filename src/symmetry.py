from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SymmetryAxisAssignment:
    label: str
    nearest_order: int | None
    distance_degrees: float | None

    @property
    def nearest_label(self):
        return (
            f"{self.nearest_order}-fold"
            if self.nearest_order is not None
            else "general"
        )


def assign_symmetry_axis(view_direction, symmetry, *, threshold_degrees=5):
    """Assign an icosahedral View Direction to its nearest named axis."""
    group_name = symmetry.strip().upper()
    if group_name in {"I1", "I2"}:
        group_name = "I"
    if group_name != "I":
        return SymmetryAxisAssignment("general", None, None)

    direction = np.asarray(view_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    rotation_vectors = Rotation.create_group("I").as_rotvec()
    angles = np.linalg.norm(rotation_vectors, axis=1)
    candidates = []
    for order, target_angle in ((2, np.pi), (3, 2 * np.pi / 3), (5, 2 * np.pi / 5)):
        selected = np.isclose(angles, target_angle, atol=1e-7)
        axes = rotation_vectors[selected] / angles[selected, None]
        best_dot = np.max(np.abs(axes @ direction))
        distance = float(np.rad2deg(np.arccos(np.clip(best_dot, -1, 1))))
        if distance < 1e-7:
            distance = 0.0
        candidates.append((distance, order))

    distance, order = min(candidates)
    label = f"{order}-fold" if distance <= threshold_degrees else "general"
    return SymmetryAxisAssignment(label, order, distance)
