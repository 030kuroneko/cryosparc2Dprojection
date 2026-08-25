"""Exact symmetry-axis reference projections.

This module is intentionally small: it is the projection seam used by the
axis search and by the live convention fixture.  The array named ``projection``
is always the unsharpened Matching Map result.  ``display_projection`` is only
the CryoSPARC vertical display normalization and never feeds scoring.
"""

from dataclasses import dataclass

import numpy as np

from cryosparc_2d_projection.axis_registry import AxisFamilyRecord, get_axis_family
from cryosparc_2d_projection.projection import project_volume_at_rotation


@dataclass(frozen=True)
class AxisReferenceProjection:
    """Raw and display forms of one exact axis reference projection."""

    family: AxisFamilyRecord
    projection: np.ndarray
    display_projection: np.ndarray
    rotation_matrix: np.ndarray
    pixel_size_A: float
    source_map: str = "matching"

    def __post_init__(self):
        projection = np.asarray(self.projection, dtype=np.float32)
        display_projection = np.asarray(self.display_projection, dtype=np.float32)
        rotation_matrix = np.asarray(self.rotation_matrix, dtype=float)
        if projection.ndim != 2 or projection.shape[0] != projection.shape[1]:
            raise ValueError("axis reference projection must be a square 2D array")
        if display_projection.shape != projection.shape:
            raise ValueError("display projection must have the raw projection shape")
        if rotation_matrix.shape != (3, 3):
            raise ValueError("axis reference camera must be 3 x 3")
        projection = projection.copy()
        display_projection = display_projection.copy()
        rotation_matrix = rotation_matrix.copy()
        projection.setflags(write=False)
        display_projection.setflags(write=False)
        rotation_matrix.setflags(write=False)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "display_projection", display_projection)
        object.__setattr__(self, "rotation_matrix", rotation_matrix)
        object.__setattr__(self, "pixel_size_A", float(self.pixel_size_A))
        if self.pixel_size_A <= 0 or not np.isfinite(self.pixel_size_A):
            raise ValueError("axis reference pixel size must be positive and finite")
        if self.source_map != "matching":
            raise ValueError("axis references must be generated from the matching map")

def project_axis_reference(
    matching_map,
    axis_family,
    *,
    pixel_size_A=1.0,
):
    """Project the unsharpened Matching Map along an exact axis.

    ``matching_map`` is deliberately the only volume argument.  A Rendering
    Map cannot accidentally influence this result.  Display normalization is
    always the same vertical flip used by static CryoSPARC Class Results; the
    raw projection remains unchanged.
    """

    family = (
        axis_family
        if isinstance(axis_family, AxisFamilyRecord)
        else get_axis_family("I", axis_family)
    )
    volume = np.asarray(matching_map, dtype=np.float32)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError("matching map must be a cubic 3D array")
    projection = project_volume_at_rotation(volume, family.canonical_camera_matrix)
    projection = np.asarray(projection, dtype=np.float32)
    display_projection = np.flipud(projection)
    return AxisReferenceProjection(
        family=family,
        projection=projection,
        display_projection=display_projection,
        rotation_matrix=family.canonical_camera_matrix,
        pixel_size_A=pixel_size_A,
    )
