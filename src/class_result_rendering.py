"""Produce one complete local Class Result set after camera selection."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import Callable

from cryosparc import mrc
import numpy as np

from cryosparc_2d_projection.auto_crop import PhysicalCameraView, compute_auto_crop_2d_framing
from cryosparc_2d_projection.matching_grid import validate_native_class_grids
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.projection import project_native_matched_projection
from cryosparc_2d_projection.scoring import (
    BandLimitedScoreConfig,
    compute_diagnostic_band_limited_score,
)
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    SurfaceRenderMemoryError,
    build_surface_model,
    get_surface_camera_viewport_A,
    resolve_surface_sampling_grid,
    write_camera_view_render,
)
from cryosparc_2d_projection.viewer import (
    create_class_preview_figure,
    create_class_preview_pages,
    write_matched_projection_thumbnail,
)


@dataclass(frozen=True)
class ClassResultInput:
    class_id: int
    class_average: object
    orientation: object
    camera: object
    search_projection: np.ndarray
    search_pixel_size_A: float

    @property
    def class_number(self):
        return self.class_id + 1


@dataclass(frozen=True)
class ClassResultRenderingEvent:
    kind: str
    stage: str
    message: str
    class_number: int | None = None
    output_name: str | None = None


@dataclass(frozen=True)
class ClassResultRenderingRequest:
    output_directory: Path
    classes: tuple[ClassResultInput, ...]
    matching_map: np.ndarray
    matching_pixel_size_A: float
    rendering_map: np.ndarray
    rendering_pixel_size_A: float
    symmetry: str
    render_options: ClassRenderOptions = field(default_factory=ClassRenderOptions)
    comparison_options: ComparisonRenderOptions = field(default_factory=ComparisonRenderOptions)
    diagnostic_score_config: BandLimitedScoreConfig = field(default_factory=BandLimitedScoreConfig)
    progress_callback: Callable[[ClassResultRenderingEvent], None] | None = None
    warning_callback: Callable[[ClassResultRenderingEvent], None] | None = None
    cryosparc_version: str = "5.0.6"


@dataclass(frozen=True)
class ClassRenderedStack:
    path: Path
    count: int
    shape: tuple[int, int]
    pixel_size_A: float


@dataclass(frozen=True)
class ClassResultSet:
    output_directory: Path
    reproducibility_metadata: dict[str, object]
    reproducibility_metadata_path: Path
    stacks: dict[str, ClassRenderedStack]
    render_paths: dict[int, Path]
    comparison_paths: dict[int, Path]
    thumbnail_path: Path
    preview_paths: tuple[Path, ...]
    warnings: tuple[ClassResultRenderingEvent, ...] = field(default_factory=tuple)


class ClassResultRenderingError(RuntimeError):
    """A complete local Class Result set could not be produced."""


class NativeReprojectionError(ClassResultRenderingError):
    """A native Matched Projection could not be generated for a class."""


_MANAGED_RESULT_NAMES = (
    "class_orientations.json",
    "class_projections.mrcs",
    "search_projections.mrcs",
    "renders",
)


def render_class_results(request: ClassResultRenderingRequest) -> ClassResultSet:
    """Produce one all-or-nothing local Class Result set."""
    classes = _validate_request(request)
    output_directory = Path(request.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_directory,
        prefix=".class-result-staging-",
    ) as staging_text:
        staging_directory = Path(staging_text)
        rendered = _render_to_staging(request, classes, staging_directory)
        _promote_complete_result(staging_directory, output_directory)
    return ClassResultSet(
        output_directory=output_directory,
        reproducibility_metadata=rendered.reproducibility_metadata,
        reproducibility_metadata_path=output_directory / "class_orientations.json",
        stacks={
            name: ClassRenderedStack(
                output_directory / stack.path.name,
                stack.count,
                stack.shape,
                stack.pixel_size_A,
            )
            for name, stack in rendered.stacks.items()
        },
        render_paths={
            class_id: output_directory / "renders" / path.name
            for class_id, path in rendered.render_paths.items()
        },
        comparison_paths={
            class_id: output_directory / "renders" / path.name
            for class_id, path in rendered.comparison_paths.items()
        },
        thumbnail_path=output_directory / "renders" / rendered.thumbnail_path.name,
        preview_paths=tuple(
            output_directory / "renders" / path.name
            for path in rendered.preview_paths
        ),
        warnings=rendered.warnings,
    )


def _validate_request(request):
    if not isinstance(request, ClassResultRenderingRequest):
        raise TypeError("request must be a ClassResultRenderingRequest")
    if not request.classes:
        raise ClassResultRenderingError("Class Result Rendering requires at least one Class Result")
    classes = tuple(sorted(request.classes, key=lambda item: item.class_id))
    class_ids = tuple(item.class_id for item in classes)
    if len(set(class_ids)) != len(class_ids):
        raise ClassResultRenderingError("Class Result Class Numbers must be unique")
    validate_native_class_grids(
        {item.class_id: item.class_average for item in classes},
        class_ids,
    )
    return classes


def _render_to_staging(request, classes, directory):
    from matplotlib import pyplot as plt

    warnings = []
    class_averages = {item.class_id: item.class_average for item in classes}
    orientations = {item.class_id: item.orientation for item in classes}
    cameras = {item.class_id: item.camera for item in classes}
    resolved = request.comparison_options.resolve(
        class_count=len(classes), requested_render_size=request.render_options.image_size
    )
    for message in resolved.warnings:
        _emit(request, warnings, "warning", "presentation", message)
    reproducibility_metadata = {
        "cryosparc_version": request.cryosparc_version,
        "symmetry": request.symmetry,
        "classes": [
            {
                "class_id": item.class_id,
                "class_number": item.class_number,
                "particle_count": item.orientation.particle_count,
                "view_direction": (item.orientation.view_direction.tolist()
                                   if item.orientation.view_direction is not None else None),
                "angular_spread_degrees": item.orientation.angular_spread_degrees,
            }
            for item in classes
        ],
        "presentation": {
            "comparison_dpi": resolved.comparison_dpi,
            "preview_page_size": resolved.preview_page_size,
            "requested_render_size": resolved.requested_render_size,
            "effective_render_size": resolved.effective_render_size,
            "render_size_was_automatic": resolved.render_size_was_automatic,
            "estimated_page_width_px": resolved.estimated_page_width_px,
            "estimated_page_height_px": resolved.estimated_page_height_px,
            "estimated_page_rgba_memory_bytes": resolved.estimated_page_rgba_memory_bytes,
            "warnings": list(resolved.warnings),
        },
    }
    if request.comparison_options.auto_crop_2d:
        reproducibility_metadata["presentation"]["auto_crop_2d"] = {
            "enabled": True,
            "mode": "physical_camera_fov",
        }

    native_results = {}
    diagnostics = {}
    for item in classes:
        try:
            native = project_native_matched_projection(
                item.class_average.image,
                request.matching_map,
                item.camera.rotation_matrix,
                class_pixel_size=item.class_average.pixel_size_A,
                volume_pixel_size=request.matching_pixel_size_A,
            )
        except (MemoryError, RuntimeError, ValueError) as error:
            raise NativeReprojectionError(
                f"Native Matched Projection failed for Class {item.class_number}; "
                "the bounded Search Projection was not substituted. "
                f"Cause: {error}"
            ) from error
        native_results[item.class_id] = native
        diagnostics[item.class_id] = compute_diagnostic_band_limited_score(
            item.class_average.image,
            native.matched_projection,
            pixel_size_A=item.class_average.pixel_size_A,
            settings=request.diagnostic_score_config,
        )

    sampling_grid = resolve_surface_sampling_grid(
        request.rendering_map.shape,
        request.render_options.grid_size,
    )
    _emit(
        request,
        warnings,
        "progress",
        "surface-sampling",
        _sampling_grid_message(sampling_grid, resolved.effective_render_size),
    )
    for message in sampling_grid.warnings:
        _emit(request, warnings, "warning", "surface-sampling", message)
    try:
        surface = build_surface_model(
            request.rendering_map,
            surface_level=request.render_options.surface_level,
            sampling_grid=sampling_grid,
        )
    except SurfaceRenderMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise ClassResultRenderingError(
            f"Class Result Rendering surface failed: {error}"
        ) from error
    _emit(
        request,
        warnings,
        "progress",
        "surface-rendering",
        f"Surface Level: {surface.surface_level:.6g}",
    )
    if surface.warning:
        _emit(request, warnings, "warning", "surface-rendering", surface.warning)
    reproducibility_metadata["rendering"] = {
        "map": request.render_options.map_name,
        "surface_level": surface.surface_level,
        "surface_level_was_automatic": surface.surface_level_was_automatic,
        "warning": surface.warning,
        "background": request.render_options.background,
        "image_size": resolved.effective_render_size,
        "grid_size": sampling_grid.effective_grid_size,
        **sampling_grid.as_dict(),
    }

    camera_viewport_A = None
    camera_viewport_error = None
    if request.comparison_options.auto_crop_2d:
        try:
            camera_viewport_A = get_surface_camera_viewport_A(
                surface, rendering_pixel_size_A=request.rendering_pixel_size_A
            )
            camera_viewport_A = PhysicalCameraView(
                camera_viewport_A=camera_viewport_A
            ).camera_viewport_A
        except (TypeError, ValueError, OverflowError) as error:
            camera_viewport_error = error
            _emit(
                request,
                warnings,
                "warning",
                "auto-cropped-2d-framing",
                "WARNING: Auto-Cropped 2D Framing fell back for all classes: "
                f"invalid physical camera viewport ({error})",
            )
        else:
            reproducibility_metadata["presentation"]["auto_crop_2d"][
                "camera_viewport_A"
            ] = float(camera_viewport_A)

    render_directory = directory / "renders"
    render_paths = {}
    framing_decisions = {}
    for class_entry, item in zip(
        reproducibility_metadata["classes"], classes, strict=True
    ):
        native = native_results[item.class_id]
        diagnostic = diagnostics[item.class_id]
        diagnostic_metadata = {
            key: value
            for key, value in diagnostic.metadata.items()
            if key not in {"band_limited_score_valid", "band_limited_invalid_reason"}
        }
        class_entry["camera"] = {
            "orientation_method": item.camera.orientation_method,
            "search_metadata": item.camera.search_metadata,
            "alternative_orientations": list(item.camera.alternative_orientations),
            "rotation_matrix": item.camera.rotation_matrix.tolist(),
            "quaternion_xyzw": item.camera.quaternion_xyzw.tolist(),
            "view_direction": item.camera.view_direction.tolist(),
            "in_plane_rotation_degrees": item.camera.in_plane_rotation_degrees,
            "projection_shift_pixels": native.projection_shift_pixels.tolist(),
            "search_projection_shift_pixels": item.camera.projection_shift_pixels.tolist(),
            "match_score": item.camera.match_score,
            "second_best_score": item.camera.second_best_score,
            "score_margin": item.camera.score_margin,
            "match_confidence": item.camera.match_confidence,
            "search_score_provenance": {
                "source": "bounded_search_projection",
                "role": "camera_selection_and_ranking",
                "reported_fields": [
                    "match_score",
                    "second_best_score",
                    "score_margin",
                    "match_confidence",
                ],
            },
            "diagnostic_band_limited_score": {
                "score": diagnostic.score,
                "valid": diagnostic.valid,
                "invalid_reason": diagnostic.invalid_reason,
                **diagnostic_metadata,
            },
            "search_box_size": int(item.search_projection.shape[0]),
            "search_pixel_size_A": item.search_pixel_size_A,
            "matching_box_size": int(item.class_average.image.shape[0]),
            "matching_pixel_size_A": item.class_average.pixel_size_A,
            "search_evaluation_count": item.camera.search_evaluation_count,
            "coordinate_convention": (
                "right-handed Cartesian active rotation; "
                "image rows increase downward"
            ),
        }
        render_paths[item.class_id] = write_camera_view_render(
            render_directory,
            surface=surface,
            rotation_matrix=item.camera.rotation_matrix,
            class_number=item.class_number,
            image_size=resolved.effective_render_size,
            background=request.render_options.background,
        )
        if request.comparison_options.auto_crop_2d:
            camera_view = None
            if camera_viewport_error is None:
                camera_view = PhysicalCameraView(
                    camera_viewport_A=camera_viewport_A,
                    projection_shift_pixels=tuple(native.projection_shift_pixels),
                )
            decision = compute_auto_crop_2d_framing(
                native.matched_projection.shape,
                item.class_average.pixel_size_A,
                [] if camera_view is None else [camera_view],
                enabled=True,
            )
            if decision.fallback and camera_viewport_error is None:
                _emit(
                    request,
                    warnings,
                    "warning",
                    "auto-cropped-2d-framing",
                    "WARNING: Auto-Cropped 2D Framing fell back for "
                    f"Class {item.class_number}: {decision.fallback_reason}",
                    class_number=item.class_number,
                )
            framing_decisions[item.class_id] = decision
            framing_metadata = decision.as_dict()
            framing_metadata["camera_view"] = None if camera_view is None else camera_view.as_dict()
            class_entry["presentation"] = {"auto_crop_2d": framing_metadata}

    projections = np.asarray(
        [native_results[item.class_id].matched_projection for item in classes],
        dtype=np.float32,
    )
    search_projections = np.asarray([item.search_projection for item in classes], dtype=np.float32)
    reproducibility_metadata_path = directory / "class_orientations.json"
    reproducibility_metadata_path.write_text(
        json.dumps(reproducibility_metadata, indent=2) + "\n"
    )
    matched_path = directory / "class_projections.mrcs"
    search_path = directory / "search_projections.mrcs"
    matched_pixel_size = float(classes[0].class_average.pixel_size_A)
    search_pixel_size = float(classes[0].search_pixel_size_A)
    mrc.write(matched_path, projections, matched_pixel_size)
    mrc.write(search_path, search_projections, search_pixel_size)
    stacks = {
        "matched_projections": ClassRenderedStack(
            matched_path,
            len(projections),
            tuple(projections.shape[1:]),
            matched_pixel_size,
        ),
        "search_projections": ClassRenderedStack(
            search_path,
            len(search_projections),
            tuple(search_projections.shape[1:]),
            search_pixel_size,
        ),
    }
    _emit(request, warnings, "progress", "preview-writing", "Writing result previews")
    preview_paths = []
    for page_number, page in enumerate(
        create_class_preview_pages(
            class_averages,
            projections,
            cameras,
            orientations,
            render_paths,
            diagnostic_scores=diagnostics,
            comparison_options=request.comparison_options,
            auto_crop_decisions=framing_decisions,
        ),
        start=1,
    ):
        path = render_directory / f"class_preview_{page_number:03d}.png"
        try:
            page.savefig(
                path,
                dpi=request.comparison_options.dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
        finally:
            plt.close(page)
        preview_paths.append(path)
    _emit(request, warnings, "progress", "surface-rendering", "Generating class comparison images")
    comparison_paths = {}
    for item in classes:
        path = render_directory / f"class_{item.class_number:03d}_comparison.png"
        comparison = create_class_preview_figure(
            class_averages,
            projections,
            cameras,
            orientations,
            render_paths,
            diagnostic_scores=diagnostics,
            comparison_options=request.comparison_options,
            class_ids=[item.class_id],
            auto_crop_decisions=framing_decisions,
        )
        try:
            comparison.savefig(path, dpi=request.comparison_options.dpi)
        finally:
            plt.close(comparison)
        comparison_paths[item.class_id] = path
        _emit(request, warnings, "progress", "class-completed",
              f"Class {item.class_number} result ready", class_number=item.class_number)
    thumbnail_path = write_matched_projection_thumbnail(
        render_directory / "matched_projections_thumbnail.png", projections[0]
    )
    return ClassResultSet(
        output_directory=directory,
        reproducibility_metadata=reproducibility_metadata,
        reproducibility_metadata_path=reproducibility_metadata_path,
        stacks=stacks,
        render_paths=render_paths,
        comparison_paths=comparison_paths,
        thumbnail_path=thumbnail_path,
        preview_paths=tuple(preview_paths),
        warnings=tuple(warnings),
    )


def _sampling_grid_message(sampling_grid, render_size):
    original_shape = " x ".join(str(size) for size in sampling_grid.original_shape)
    sampled_shape = " x ".join(str(size) for size in sampling_grid.sampled_shape)
    requested_grid = (
        "native"
        if sampling_grid.requested_grid_size is None
        else str(sampling_grid.requested_grid_size)
    )
    return (
        f"Surface Sampling Grid: original={original_shape}; requested={requested_grid}; "
        f"effective={sampled_shape}; mode={sampling_grid.mode}; "
        f"downsampled={'yes' if sampling_grid.was_downsampled else 'no'}; "
        f"estimated minimum working memory={sampling_grid.estimated_memory_gib:.3f} GiB "
        "(mesh and plotting allocations excluded); "
        f"Camera View Render={render_size} px."
    )


def _emit(request, warnings, kind, stage, message, *, class_number=None, output_name=None):
    event = ClassResultRenderingEvent(kind, stage, message, class_number, output_name)
    if kind == "warning":
        warnings.append(event)
        callback = request.warning_callback
    else:
        callback = request.progress_callback
    if callback is not None:
        try:
            callback(event)
        except Exception:
            pass


def _promote_complete_result(staging_directory, output_directory):
    entries = tuple(staging_directory.iterdir())
    managed_targets = [
        output_directory / name
        for name in _MANAGED_RESULT_NAMES
        if (output_directory / name).exists()
    ]
    with tempfile.TemporaryDirectory(
        dir=output_directory,
        prefix=".class-result-backup-",
    ) as backup_text:
        backup_directory = Path(backup_text)
        backups = []
        promoted = []
        try:
            for target in managed_targets:
                backup = backup_directory / target.name
                target.replace(backup)
                backups.append(backup)
            for entry in entries:
                target = output_directory / entry.name
                entry.replace(target)
                promoted.append(target)
        except Exception as error:
            for target in reversed(promoted):
                target.replace(staging_directory / target.name)
            for backup in backups:
                backup.replace(output_directory / backup.name)
            raise ClassResultRenderingError(
                f"Class Result Rendering promotion failed: {error}"
            ) from error
