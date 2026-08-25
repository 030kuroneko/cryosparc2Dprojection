from dataclasses import dataclass
from pathlib import Path
from numbers import Integral

import numpy as np
from scipy.ndimage import generate_binary_structure, label, zoom
from skimage.measure import marching_cubes


_BYTES_PER_GIB = 1024**3
# The lower-bound estimate covers the float32 sampled volume, the binary
# occupancy mask, the int32 connected-component labels, a retained-component
# mask, and the density copy used when small components are removed. Mesh
# storage is deliberately excluded because it depends on the contour.
_SURFACE_MEMORY_BYTES_PER_VOXEL = 4 + 1 + 4 + 1 + 4
_SURFACE_GRID_TIERS = (512, 384, 256, 192, 128)


def recommend_lower_surface_grid_size(failed_grid_size):
    """Return the next explicit grid to try after a memory failure.

    The standard tiers are used whenever the failed effective grid is above
    128. Smaller grids have no standard tier, so halve the failed value while
    respecting the minimum valid grid size of two. A failed grid of two has
    no valid smaller recommendation and returns ``None``.
    """
    if isinstance(failed_grid_size, bool) or not isinstance(
        failed_grid_size, Integral
    ):
        raise ValueError("failed render grid size must be an integer")
    failed_grid_size = int(failed_grid_size)
    if failed_grid_size < 2:
        raise ValueError("failed render grid size must be at least 2")
    for tier in _SURFACE_GRID_TIERS:
        if failed_grid_size > tier:
            return tier
    if failed_grid_size == 2:
        return None
    return max(2, failed_grid_size // 2)


class SurfaceRenderMemoryError(MemoryError):
    """Actionable, non-retrying memory failure for surface presentation."""

    def __init__(self, *, stage, sampling_grid, cause=None):
        self.stage = str(stage)
        self.sampling_grid = sampling_grid
        if sampling_grid is None:
            self.sampled_shape = None
            self.effective_grid_size = None
            self.recommended_grid_size = None
        else:
            self.sampled_shape = tuple(sampling_grid.sampled_shape)
            self.effective_grid_size = int(sampling_grid.effective_grid_size)
            self.recommended_grid_size = recommend_lower_surface_grid_size(
                self.effective_grid_size
            )

        if self.sampled_shape is None:
            shape_text = "unavailable"
        else:
            shape_text = " x ".join(str(size) for size in self.sampled_shape)
        if self.recommended_grid_size is None:
            recommendation = (
                "No smaller valid grid exists; the minimum render grid is 2."
            )
        else:
            recommendation = (
                "Retry explicitly with "
                f"--render-grid-size {self.recommended_grid_size} or smaller."
            )
        message = (
            f"Surface {self.stage} ran out of memory at effective sampled shape "
            f"{shape_text}. {recommendation} The operation did not retry or "
            "lower the grid automatically."
        )
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True)
class ResolvedSurfaceSamplingGrid:
    """Resolved sampling policy for extracting a rendering surface.

    ``requested_grid_size`` is the user's maximum side length, or ``None``
    when native-grid automatic mode was requested. ``sampled_shape`` is the
    actual three-dimensional array shape passed to surface extraction.
    """

    original_shape: tuple[int, int, int]
    requested_grid_size: int | None
    effective_grid_size: int
    sampled_shape: tuple[int, int, int]
    grid_size_was_automatic: bool
    was_downsampled: bool
    estimated_memory_bytes: int
    warnings: tuple[str, ...] = ()

    @property
    def mode(self):
        return "automatic" if self.grid_size_was_automatic else "manual"

    @property
    def warning(self):
        return self.warnings[0] if self.warnings else None

    @property
    def estimated_memory_gib(self):
        return self.estimated_memory_bytes / _BYTES_PER_GIB

    def as_dict(self):
        """Return JSON-compatible metadata for Job Log/result artifacts."""
        return {
            "original_shape": list(self.original_shape),
            "requested_grid_size": self.requested_grid_size,
            "effective_grid_size": self.effective_grid_size,
            "sampled_shape": list(self.sampled_shape),
            "grid_size_was_automatic": self.grid_size_was_automatic,
            "mode": self.mode,
            "was_downsampled": self.was_downsampled,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "estimated_memory_gib": self.estimated_memory_gib,
            "warnings": list(self.warnings),
        }


def resolve_surface_sampling_grid(
    volume_shape,
    requested_grid_size=None,
):
    """Resolve native or manually bounded sampling for a 3D rendering map.

    A missing request means use the complete native map grid. A manual value
    is a maximum side length only: values larger than the map never upsample,
    while smaller values downsample every axis by one common factor so the
    map's aspect ratio is preserved.
    """
    original_shape = _validate_volume_shape(volume_shape)
    if requested_grid_size is not None:
        if isinstance(requested_grid_size, bool) or not isinstance(
            requested_grid_size, Integral
        ):
            raise ValueError("render grid size must be an integer at least 2")
        requested_grid_size = int(requested_grid_size)
        if requested_grid_size < 2:
            raise ValueError("render grid size must be at least 2")

    original_max = max(original_shape)
    if requested_grid_size is None:
        sampled_shape = original_shape
    else:
        scale = min(1.0, requested_grid_size / original_max)
        sampled_shape = tuple(
            max(2, int(np.rint(dimension * scale)))
            for dimension in original_shape
        )
        # A dimension of two must remain two after integer rounding, while no
        # axis can ever exceed its native size.
        sampled_shape = tuple(
            min(native, sampled)
            for native, sampled in zip(original_shape, sampled_shape)
        )

    effective_grid_size = max(sampled_shape)
    was_downsampled = sampled_shape != original_shape
    estimated_memory_bytes = _estimate_surface_memory_bytes(sampled_shape)
    warnings = ()
    if estimated_memory_bytes > _BYTES_PER_GIB:
        shape_text = " x ".join(str(size) for size in sampled_shape)
        warnings = (
            "Surface Sampling Grid "
            f"{shape_text} estimates at least "
            f"{estimated_memory_bytes / _BYTES_PER_GIB:.1f} GiB of working "
            "memory (lower bound), exceeding the 1 GiB warning threshold; "
            "rendering may be memory-intensive.",
        )
    return ResolvedSurfaceSamplingGrid(
        original_shape=original_shape,
        requested_grid_size=requested_grid_size,
        effective_grid_size=effective_grid_size,
        sampled_shape=sampled_shape,
        grid_size_was_automatic=requested_grid_size is None,
        was_downsampled=was_downsampled,
        estimated_memory_bytes=estimated_memory_bytes,
        warnings=warnings,
    )


def _validate_volume_shape(volume_shape):
    try:
        dimensions = tuple(volume_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("rendering map shape must contain three dimensions") from error
    if (
        len(dimensions) != 3
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or dimension < 2
            for dimension in dimensions
        )
    ):
        raise ValueError(
            "rendering map shape must contain three dimensions of at least 2"
        )
    return tuple(int(dimension) for dimension in dimensions)


def _estimate_surface_memory_bytes(sampled_shape):
    voxel_count = int(np.prod(sampled_shape, dtype=np.int64))
    return voxel_count * _SURFACE_MEMORY_BYTES_PER_VOXEL


@dataclass(frozen=True)
class SurfaceModel:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    surface_level: float
    surface_level_was_automatic: bool = False
    warning: str | None = None
    sampling_grid: ResolvedSurfaceSamplingGrid | None = None


@dataclass(frozen=True)
class ClassRenderOptions:
    """Validated presentation policy for Class Result rendering artifacts."""

    surface_level: float | None = None
    map_name: str = "map"
    background: str = "dark"
    image_size: int | None = None
    grid_size: int | None = None

    def __post_init__(self):
        if self.map_name not in {"map", "sharpened"}:
            raise ValueError("rendering map must be 'map' or 'sharpened'")
        if self.background not in {"dark", "light"}:
            raise ValueError("render background must be 'dark' or 'light'")
        if self.image_size is not None and self.image_size < 64:
            raise ValueError("render image size must be at least 64 pixels")
        if self.grid_size is not None and (
            type(self.grid_size) is not int or self.grid_size < 2
        ):
            raise ValueError("render grid size must be an integer at least 2")


def build_surface_model(volume, *, surface_level, max_size=None):
    """Extract a centered triangular isosurface from a 3D density map."""
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("rendering map must be a 3D array")
    sampling_grid = resolve_surface_sampling_grid(volume.shape, max_size)
    sampled_shape = sampling_grid.sampled_shape
    try:
        sampled = (
            volume
            if sampled_shape == volume.shape
            else zoom(
                volume,
                tuple(
                    sampled / native
                    for sampled, native in zip(sampled_shape, volume.shape)
                ),
                order=1,
                mode="nearest",
                prefilter=False,
            )
        )
    except MemoryError as error:
        raise SurfaceRenderMemoryError(
            stage="surface extraction", sampling_grid=sampling_grid
        ) from error
    surface_level_was_automatic = surface_level is None
    warning = None
    if surface_level_was_automatic:
        mean = float(sampled.mean())
        standard_deviation = float(sampled.std())
        candidates = [
            (mean + multiplier * standard_deviation, multiplier)
            for multiplier in (1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0)
        ]
    else:
        candidates = [(float(surface_level), None)]

    last_error = None
    for candidate, multiplier in candidates:
        if not sampled.min() < candidate < sampled.max():
            continue
        try:
            vertices_zyx, faces, normals_zyx = _extract_triangle_mesh(
                sampled, candidate
            )
        except MemoryError as error:
            raise SurfaceRenderMemoryError(
                stage="surface extraction", sampling_grid=sampling_grid
            ) from error
        except (RuntimeError, ValueError) as error:
            last_error = error
            if not surface_level_was_automatic:
                raise
            continue
        surface_level = candidate
        if surface_level_was_automatic and multiplier != 1.5:
            warning = (
                "Automatic Surface Level was lowered from mean + 1.5 sigma "
                f"to mean + {multiplier:g} sigma after an unusable level."
            )
        break
    else:
        message = "rendering map has no usable automatic Surface Level"
        if last_error is not None:
            raise ValueError(message) from last_error
        raise ValueError(message)

    surface_level = float(surface_level)
    vertices_xyz = vertices_zyx[:, ::-1]
    vertices_xyz -= (np.asarray(sampled.shape[::-1], dtype=float) - 1) / 2
    normals_xyz = normals_zyx[:, ::-1]
    normals_xyz /= np.linalg.norm(normals_xyz, axis=1, keepdims=True)
    return SurfaceModel(
        vertices=vertices_xyz,
        faces=np.asarray(faces, dtype=np.int32),
        normals=normals_xyz,
        surface_level=surface_level,
        surface_level_was_automatic=surface_level_was_automatic,
        warning=warning,
        sampling_grid=sampling_grid,
    )


def _extract_triangle_mesh(sampled, surface_level):
    occupied = sampled >= surface_level
    component_labels, component_count = label(
        occupied, structure=generate_binary_structure(3, 3)
    )
    if component_count:
        sizes = np.bincount(component_labels.ravel())[1:]
        minimum_size = max(1.0, float(sizes.max()) * 0.01)
        retained_labels = np.flatnonzero(sizes >= minimum_size) + 1
        retained = np.isin(component_labels, retained_labels)
        sampled = sampled.copy()
        sampled[occupied & ~retained] = sampled.min()

    vertices, faces, normals, _ = marching_cubes(sampled, level=surface_level)
    return vertices, faces, normals


def write_camera_view_render(
    output_directory,
    *,
    surface,
    rotation_matrix,
    class_number,
    image_size=1024,
    background="dark",
):
    """Save the exact orthographic Camera View Render for one class."""
    output_directory = Path(output_directory)
    camera_view_path = output_directory / f"class_{class_number:03d}_exact.png"
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_surface_image(
            camera_view_path,
            surface,
            np.asarray(rotation_matrix, dtype=float),
            image_size=image_size,
            background=background,
        )
    except MemoryError as error:
        raise SurfaceRenderMemoryError(
            stage="PNG rendering",
            sampling_grid=surface.sampling_grid,
        ) from error
    return camera_view_path


def _write_surface_image(path, surface, rotation_matrix, *, image_size, background):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if image_size < 64:
        raise ValueError("render size must be at least 64 pixels")
    if background not in {"dark", "light"}:
        raise ValueError("render background must be 'dark' or 'light'")
    background_color = "#080b10" if background == "dark" else "#ffffff"

    vertices = surface.vertices @ np.asarray(rotation_matrix, dtype=float).T
    # CryoSPARC displays raw MRC row zero at the bottom. Matplotlib's Cartesian
    # +y convention already gives the Camera View Render that vertical display
    # orientation, so no horizontal or vertical mirror is applied here.
    triangles = vertices[surface.faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    face_normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.where(lengths == 0, 1.0, lengths)
    light = np.array([-0.35, -0.45, 1.0])
    light /= np.linalg.norm(light)
    # Marching-cubes winding may face inward; two-sided lighting keeps the
    # requested white surface bright while retaining depth-defining shading.
    brightness = 0.35 + 0.65 * np.abs(face_normals @ light)
    face_colors = np.column_stack(
        [brightness, brightness, brightness, np.ones(len(brightness))]
    )

    dpi = max(72.0, image_size / 8.0)
    figure_size_inches = image_size / dpi
    figure = Figure(
        figsize=(figure_size_inches, figure_size_inches),
        dpi=dpi,
        facecolor=background_color,
    )
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.02, 0.02, 0.96, 0.96), projection="3d")
    axis.set_facecolor(background_color)
    mesh = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors="none",
        linewidths=0,
        antialiased=False,
        zsort="average",
    )
    axis.add_collection3d(mesh)
    radius = max(float(np.max(np.linalg.norm(surface.vertices, axis=1))), 1.0) * 1.12
    axis.set(
        xlim=(-radius, radius),
        ylim=(-radius, radius),
        zlim=(-radius, radius),
    )
    axis.set_box_aspect((1, 1, 1), zoom=1.45)
    axis.set_proj_type("ortho")
    axis.view_init(elev=90, azim=-90)
    axis.set_axis_off()
    figure.savefig(path, dpi=dpi, facecolor=background_color)
