"""Named symmetry-axis families used by the reference-first axis search.

The registry is deliberately data driven.  The search code consumes these
records and does not need to know that the first supported convention is
icosahedral.  The numeric records are the reconstructed CryoSPARC ``I``
convention documented in ``docs/specs/symmetry-axis-class-search.md``.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AxisCanonicalPresentation:
    """The stable presentation transform associated with an axis family."""

    rule: str
    camera_matrix: np.ndarray
    roll_degrees: float = 0.0
    reference_family_name: str | None = None
    projected_axis: np.ndarray | None = None

    def __post_init__(self):
        matrix = np.asarray(self.camera_matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("canonical presentation camera must be 3 x 3")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "camera_matrix", matrix)
        if self.projected_axis is not None:
            projected_axis = np.asarray(self.projected_axis, dtype=float)
            if projected_axis.shape != (3,):
                raise ValueError("projected cross-family axis must have shape (3,)")
            projected_axis = projected_axis.copy()
            projected_axis.setflags(write=False)
            object.__setattr__(self, "projected_axis", projected_axis)
        if not np.isfinite(self.roll_degrees):
            raise ValueError("canonical presentation roll must be finite")


@dataclass(frozen=True)
class AxisFamilyRecord:
    """One named family of exact symmetry axes.

    ``canonical_camera_matrix`` uses the same complete camera convention as
    :func:`cryosparc_2d_projection.projection.project_volume_at_rotation`.
    Its third row is the representative viewing direction.  Arrays are made
    read-only so a caller cannot accidentally mutate the process-wide
    registry.
    """

    symmetry: str
    name: str
    undirected_axis_count: int
    directed_axis_count: int
    representative_view_direction: np.ndarray
    canonical_camera_matrix: np.ndarray
    roll_period_degrees: float
    canonical_presentation_rule: str = "nearest_cross_family_axis_horizontal"

    def __post_init__(self):
        symmetry = str(self.symmetry).strip().upper()
        name = _normalize_family_name(self.name)
        direction = np.asarray(self.representative_view_direction, dtype=float)
        matrix = np.asarray(self.canonical_camera_matrix, dtype=float)
        if direction.shape != (3,):
            raise ValueError("representative view direction must have shape (3,)")
        if matrix.shape != (3, 3):
            raise ValueError("canonical camera matrix must have shape (3, 3)")
        if not np.isfinite(direction).all() or not np.isfinite(matrix).all():
            raise ValueError("axis family vectors and matrices must be finite")
        direction_norm = np.linalg.norm(direction)
        if not np.isclose(direction_norm, 1.0, atol=1e-8):
            raise ValueError("representative view direction must be unit length")
        if not np.allclose(matrix[2], direction, atol=1e-8):
            raise ValueError("canonical camera third row must equal view direction")
        matrix = matrix.copy()
        direction = direction.copy()
        matrix.setflags(write=False)
        direction.setflags(write=False)
        object.__setattr__(self, "symmetry", symmetry)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "representative_view_direction", direction)
        object.__setattr__(self, "canonical_camera_matrix", matrix)
        if self.undirected_axis_count <= 0 or self.directed_axis_count <= 0:
            raise ValueError("axis counts must be positive")
        if self.roll_period_degrees <= 0 or not np.isfinite(self.roll_period_degrees):
            raise ValueError("axis roll period must be positive and finite")

    def camera_for_roll(self, roll_degrees):
        """Return the complete camera after a display/in-plane roll.

        Roll is applied in camera coordinates, so the third row (the view
        direction) remains exactly the representative axis.  The full angle
        is retained in the returned camera; periodicity is a property of the
        symmetry-equivalent volume/projection, not a modulo operation on the
        camera itself.
        """

        rolled = Rotation.from_euler("z", float(roll_degrees), degrees=True).as_matrix() @ self.canonical_camera_matrix
        rolled = np.asarray(rolled, dtype=float)
        rolled.setflags(write=False)
        return rolled

    def canonical_presentation(self, *, roll_degrees=0.0):
        """Return the deterministic display rule for this family.

        Manual roll is intentionally represented separately from the registry
        record.  It is a display-only transform and therefore cannot mutate
        the exact-axis camera or affect scoring.
        """

        reference_family, projected_axis = _nearest_cross_family_axis(self)
        image_y = self.canonical_camera_matrix[1]
        if abs(float(projected_axis @ image_y)) > 1e-8:
            raise ValueError(
                f"canonical {self.name} camera does not make its nearest "
                "cross-family axis horizontal"
            )
        return AxisCanonicalPresentation(
            rule=self.canonical_presentation_rule,
            camera_matrix=self.canonical_camera_matrix,
            roll_degrees=float(roll_degrees),
            reference_family_name=reference_family.name,
            projected_axis=projected_axis,
        )


def _normalize_family_name(value):
    normalized = str(value).strip().lower()
    if normalized not in {"2fold", "3fold", "5fold"}:
        raise ValueError("axis family must be one of 2fold, 3fold, 5fold")
    return normalized


class AxisFamilyRegistry:
    """Immutable lookup registry for one supported symmetry convention."""

    def __init__(self, symmetry, records):
        symmetry = str(symmetry).strip().upper()
        records = tuple(records)
        if not records:
            raise ValueError("an axis-family registry must contain at least one record")
        if any(record.symmetry != symmetry for record in records):
            raise ValueError("all axis-family records must use the registry symmetry")
        by_name = {record.name: record for record in records}
        if len(by_name) != len(records):
            raise ValueError("axis-family names must be unique")
        self.symmetry = symmetry
        self._records = records
        self._by_name = MappingProxyType(by_name)

    @classmethod
    def for_symmetry(cls, symmetry):
        normalized = str(symmetry).strip().upper()
        if normalized != "I":
            raise ValueError("axis-family search only supports I")
        return ICOSAHEDRAL_AXIS_FAMILY_REGISTRY

    def records(self):
        return self._records

    def lookup(self, family):
        normalized = _normalize_family_name(family)
        try:
            return self._by_name[normalized]
        except KeyError as error:
            available = ", ".join(record.name for record in self._records)
            raise ValueError(
                f"unknown axis family {family!r}; choose one of {available}"
            ) from error

_THREEFOLD_X_COMPONENT = 0.934172358962716
_THREEFOLD_Y_COMPONENT = 0.356822089773090
_FIVEFOLD_X_COMPONENT = 0.850650808352040
_FIVEFOLD_Z_COMPONENT = 0.525731112119134

ICOSAHEDRAL_AXIS_FAMILY_REGISTRY = AxisFamilyRegistry(
    "I",
    (
        AxisFamilyRecord(
            symmetry="I",
            name="2fold",
            undirected_axis_count=15,
            directed_axis_count=30,
            representative_view_direction=np.array([0.0, 1.0, 0.0]),
            canonical_camera_matrix=np.array(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            roll_period_degrees=180.0,
        ),
        AxisFamilyRecord(
            symmetry="I",
            name="3fold",
            undirected_axis_count=10,
            directed_axis_count=20,
            representative_view_direction=np.array(
                [-_THREEFOLD_X_COMPONENT, _THREEFOLD_Y_COMPONENT, 0.0]
            ),
            canonical_camera_matrix=np.array(
                [
                    [_THREEFOLD_Y_COMPONENT, _THREEFOLD_X_COMPONENT, 0.0],
                    [0.0, 0.0, -1.0],
                    [-_THREEFOLD_X_COMPONENT, _THREEFOLD_Y_COMPONENT, 0.0],
                ]
            ),
            roll_period_degrees=120.0,
        ),
        AxisFamilyRecord(
            symmetry="I",
            name="5fold",
            undirected_axis_count=6,
            directed_axis_count=12,
            representative_view_direction=np.array(
                [-_FIVEFOLD_X_COMPONENT, 0.0, _FIVEFOLD_Z_COMPONENT]
            ),
            canonical_camera_matrix=np.array(
                [
                    [_FIVEFOLD_Z_COMPONENT, 0.0, _FIVEFOLD_X_COMPONENT],
                    [0.0, 1.0, 0.0],
                    [-_FIVEFOLD_X_COMPONENT, 0.0, _FIVEFOLD_Z_COMPONENT],
                ]
            ),
            roll_period_degrees=72.0,
        ),
    ),
)

def _nearest_cross_family_axis(record):
    """Return the nearest directed axis from another registered family."""

    group = Rotation.create_group(record.symmetry).as_matrix()
    candidates = []
    for other in ICOSAHEDRAL_AXIS_FAMILY_REGISTRY.records():
        if other.name == record.name:
            continue
        # Vectors are rows throughout the pose/camera code.  Include both
        # directions because an undirected symmetry axis has two poles.
        directed = np.concatenate(
            [
                np.asarray(
                    [other.representative_view_direction @ operator for operator in group]
                ),
                np.asarray(
                    [-other.representative_view_direction @ operator for operator in group]
                ),
            ]
        )
        for axis in directed:
            alignment = abs(float(axis @ record.representative_view_direction))
            view = record.representative_view_direction
            projected = axis - float(axis @ view) * view
            norm = np.linalg.norm(projected)
            if norm == 0:
                continue
            candidates.append(
                (
                    alignment,
                    other,
                    axis,
                    projected / norm,
                )
            )
    if not candidates:
        raise ValueError("an axis family requires another family for presentation")
    highest_alignment = max(item[0] for item in candidates)
    nearest = [
        item
        for item in candidates
        if np.isclose(item[0], highest_alignment, atol=1e-8, rtol=0)
    ]
    image_y = record.canonical_camera_matrix[1]
    nearest.sort(
        key=lambda item: (
            abs(float(item[3] @ image_y)),
            item[1].name,
            tuple(np.round(item[3], 12)),
        )
    )
    return nearest[0][1], nearest[0][3]


def axis_family_records(symmetry="I"):
    """Return records in their deterministic family order."""

    return AxisFamilyRegistry.for_symmetry(symmetry).records()


def get_axis_family(symmetry, family=None):
    """Look up one named axis-family record in a supported registry."""

    if family is None:
        raise TypeError("get_axis_family requires symmetry and family")
    return AxisFamilyRegistry.for_symmetry(symmetry).lookup(family)
