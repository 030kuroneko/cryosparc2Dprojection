from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label, zoom
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class SurfaceModel:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    surface_level: float
    surface_level_was_automatic: bool = False
    warning: str | None = None


@dataclass(frozen=True)
class ClassRenderOptions:
    """Validated presentation policy for Class Result rendering artifacts."""

    surface_level: float | None = None
    map_name: str = "map"
    background: str = "dark"
    oblique_tilt_degrees: float = 20
    image_size: int = 1024
    grid_size: int = 192

    def __post_init__(self):
        if self.map_name not in {"map", "sharpened"}:
            raise ValueError("rendering map must be 'map' or 'sharpened'")
        if self.background not in {"dark", "light"}:
            raise ValueError("render background must be 'dark' or 'light'")
        if self.image_size < 64:
            raise ValueError("render image size must be at least 64 pixels")
        if not 2 <= self.grid_size <= 192:
            raise ValueError("render grid size must be between 2 and 192")


@dataclass(frozen=True)
class CameraRenderPaths:
    camera_view_path: Path
    oblique_inspection_path: Path


def build_surface_model(volume, *, surface_level, max_size=192):
    """Extract a centered triangular isosurface from a 3D density map."""
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("rendering map must be a 3D array")
    if max_size < 2:
        raise ValueError("render grid size must be at least 2")
    if max_size > 192:
        raise ValueError("render grid size must not exceed 192")

    scale = min(1.0, float(max_size) / max(volume.shape))
    sampled = (
        volume
        if np.isclose(scale, 1.0)
        else zoom(volume, scale, order=1, mode="nearest", prefilter=False)
    )
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


def write_camera_view_renders(
    output_directory,
    *,
    surface,
    rotation_matrix,
    class_number,
    match_score,
    match_confidence,
    symmetry_label,
    symmetry_distance_degrees,
    oblique_tilt_degrees=20,
    image_size=1024,
    background="dark",
):
    """Save a Camera View Render and Oblique Inspection Render for one class."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    exact_path = output_directory / f"class_{class_number:03d}_exact.png"
    oblique_path = output_directory / f"class_{class_number:03d}_oblique.png"
    base_rotation = np.asarray(rotation_matrix, dtype=float)
    tilt = Rotation.from_euler(
        "xy", [oblique_tilt_degrees, oblique_tilt_degrees], degrees=True
    ).as_matrix()
    symmetry_text = (
        symmetry_label
        if symmetry_distance_degrees is None
        else f"{symmetry_label} {symmetry_distance_degrees:.1f}°"
    )
    common_title = (
        f"Class {class_number}\n"
        f"score={match_score:.3f} | {match_confidence} | {symmetry_text}\n"
        f"level={surface.surface_level:.4g}"
    )
    _write_surface_image(
        exact_path,
        surface,
        base_rotation,
        title=f"Camera View Render | {common_title}",
        image_size=image_size,
        background=background,
    )
    _write_surface_image(
        oblique_path,
        surface,
        tilt @ base_rotation,
        title=f"Oblique Inspection Render | {common_title}",
        image_size=image_size,
        background=background,
    )
    return CameraRenderPaths(
        camera_view_path=exact_path,
        oblique_inspection_path=oblique_path,
    )


def _write_surface_image(path, surface, rotation_matrix, *, title, image_size, background):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if image_size < 64:
        raise ValueError("render size must be at least 64 pixels")
    if background not in {"dark", "light"}:
        raise ValueError("render background must be 'dark' or 'light'")
    background_color = "#080b10" if background == "dark" else "#ffffff"
    foreground_color = "#ffffff" if background == "dark" else "#111111"

    vertices = surface.vertices @ np.asarray(rotation_matrix, dtype=float).T
    # MRC image rows increase downward, while Matplotlib's Cartesian +y points up.
    vertices[:, 1] *= -1
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
    axis = figure.add_axes((0.04, 0.02, 0.92, 0.78), projection="3d")
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
    axis.set_box_aspect((1, 1, 1), zoom=1.35)
    axis.set_proj_type("ortho")
    axis.view_init(elev=90, azim=-90)
    axis.set_axis_off()
    axis.set_title(
        title,
        color=foreground_color,
        fontsize=max(5, min(9, image_size / 64)),
        pad=0,
    )
    figure.savefig(path, dpi=dpi, facecolor=background_color)
