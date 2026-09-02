"""Render one complete Axis Search Run into local result files."""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import tempfile
from time import monotonic
from typing import Callable

import numpy as np
from cryosparc import mrc
from PIL import Image

from cryosparc_2d_projection.axis_presentation import (
    AXIS_RESULT_COLUMNS,
    EXACT_AXIS_RESULT_COLUMNS,
    AxisResultLabel,
    AxisResultPanelRow,
    ExactAxisResultPanelRow,
    apply_axis_display_roll,
    create_axis_result_figure,
)
from cryosparc_2d_projection.auto_crop import (
    AUTO_CROP_MAX_ZOOM,
    AUTO_CROP_PADDING_FRACTION,
    compute_auto_crop_2d_framing,
)
from cryosparc_2d_projection.axis_search import (
    AxisProximityConfig,
    AxisRefinementResult,
    AxisSearchConfig,
    AxisSearchResult,
)
from cryosparc_2d_projection.matching_grid import prepare_native_matching_grid
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.projection import (
    project_native_matched_projection,
    project_volume_at_rotation,
)
from cryosparc_2d_projection.scoring import compute_diagnostic_band_limited_score
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    build_surface_model,
    get_surface_silhouette_bounds,
    resolve_surface_sampling_grid,
    write_camera_view_render,
)


class AxisResultRenderingEventCode(str, Enum):
    RESULT_RENDERING_STARTED = "result-rendering-started"
    SURFACE_SAMPLING = "surface-sampling"
    CANDIDATE_COMPLETED = "candidate-completed"
    OUTPUT_WRITING_STARTED = "output-writing-started"
    OUTPUT_WRITING_COMPLETED = "output-writing-completed"
    WARNING = "warning"


@dataclass(frozen=True)
class AxisResultRenderingEvent:
    code: AxisResultRenderingEventCode
    kind: str
    stage: str
    message: str
    family_name: str | None = None
    class_number: int | None = None
    output_name: str | None = None


@dataclass(frozen=True)
class AxisRenderedStack:
    filename: str
    data: np.ndarray
    pixel_size_A: float


@dataclass(frozen=True)
class AxisResultRenderingRequest:
    output_directory: Path
    search_result: AxisSearchResult
    refinement: AxisRefinementResult | None
    matching_map: np.ndarray
    rendering_map: np.ndarray
    class_pixel_size_A: float
    map_pixel_size_A: float
    config: AxisSearchConfig
    proximity_config: AxisProximityConfig
    axis_rolls: dict[str, float]
    comparison_options: ComparisonRenderOptions
    render_options: ClassRenderOptions
    refine_near_axis: bool
    timings: dict[str, dict[str, float]]
    progress_callback: Callable[[AxisResultRenderingEvent], None] | None = None
    warning_callback: Callable[[AxisResultRenderingEvent], None] | None = None
    clock: Callable[[], float] = monotonic
    run_started_at: float | None = None


@dataclass(frozen=True)
class AxisResultRenderingResult:
    artifact: dict[str, object]
    stacks: dict[str, AxisRenderedStack]
    preview_pages: tuple[object, ...]
    preview_path: Path
    warnings: tuple[AxisResultRenderingEvent, ...] = field(default_factory=tuple)


class AxisResultRenderingError(RuntimeError):
    """A complete Result Rendering result set could not be produced."""


_MANAGED_RESULT_NAMES = (
    "axis_candidates_raw.mrcs",
    "axis_candidates_aligned.mrcs",
    "axis_exact_references.mrcs",
    "axis_exact_search_projections.mrcs",
    "axis_exact_matched_projections.mrcs",
    "axis_search_preview.mrcs",
    "axis_near_projections.mrcs",
    "axis_near_search_projections.mrcs",
    "axis_near_matched_projections.mrcs",
    "axis_search_results.json",
    "renders",
)


def render_axis_search_results(request: AxisResultRenderingRequest):
    """Write one complete Axis Search Run result set."""

    try:
        if request.refine_near_axis and request.refinement is None:
            candidate = request.search_result.rows[0]
            raise _candidate_output_error(
                candidate,
                "near-axis-result",
                ValueError("near-axis mode requires refinement results"),
            )
        if not request.refine_near_axis and request.refinement is not None:
            candidate = request.search_result.rows[0]
            raise _candidate_output_error(
                candidate,
                "near-axis-result",
                ValueError("refinement results require near-axis mode"),
            )
        if request.refinement is not None and len(request.refinement.rows) != len(
            request.search_result.rows
        ):
            candidate = request.search_result.rows[0]
            raise _candidate_output_error(
                candidate,
                "near-axis-result",
                ValueError("refinement rows must match every search candidate"),
            )
        if request.refinement is not None:
            for candidate, refined in zip(
                request.search_result.rows,
                request.refinement.rows,
            ):
                refined_candidate = refined.exact_candidate
                if (
                    refined_candidate.family_name != candidate.family_name
                    or refined_candidate.class_number != candidate.class_number
                ):
                    raise _candidate_output_error(
                        candidate,
                        "near-axis-result",
                        ValueError("refinement row belongs to another candidate"),
                    )
        if request.search_result.rows:
            native_shape = request.search_result.rows[0].raw_class.shape
            for candidate in request.search_result.rows[1:]:
                if candidate.raw_class.shape != native_shape:
                    raise _candidate_output_error(
                        candidate,
                        "axis_candidates_raw",
                        ValueError(
                            "native box size differs: "
                            f"expected {native_shape}, got {candidate.raw_class.shape}"
                        ),
                    )
        output_directory = Path(request.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        warnings = []
        with tempfile.TemporaryDirectory(
            dir=output_directory,
            prefix=".axis-result-rendering-",
        ) as temporary_directory:
            staging_directory = Path(temporary_directory)
            result = _render_to_staging(request, staging_directory, warnings)
            _promote_complete_result(staging_directory, output_directory, request)
            _emit(
                request,
                AxisResultRenderingEventCode.OUTPUT_WRITING_COMPLETED,
                "progress",
                "output-writing",
                "completed",
            )
    except AxisResultRenderingError:
        raise
    except Exception as error:
        raise _output_error(request, "local-result-set", error) from error
    return AxisResultRenderingResult(
        artifact=result.artifact,
        stacks=result.stacks,
        preview_pages=result.preview_pages,
        preview_path=output_directory / result.preview_path.name,
        warnings=tuple(warnings),
    )


def _render_to_staging(request, directory, warnings):
    timings = {name: dict(value) for name, value in request.timings.items()}
    render_started_at = request.clock()
    _emit(
        request,
        AxisResultRenderingEventCode.RESULT_RENDERING_STARTED,
        "progress",
        "result-rendering",
        "started",
    )
    sampling_grid = resolve_surface_sampling_grid(
        request.rendering_map.shape,
        request.render_options.grid_size,
    )
    _emit(
        request,
        AxisResultRenderingEventCode.SURFACE_SAMPLING,
        "progress",
        "result-rendering",
        _sampling_grid_message(sampling_grid),
    )
    for warning in sampling_grid.warnings:
        event = AxisResultRenderingEvent(
            AxisResultRenderingEventCode.WARNING,
            "warning",
            "result-rendering",
            warning,
        )
        warnings.append(event)
        _notify(request.warning_callback, event)
    try:
        surface = build_surface_model(
            request.rendering_map,
            surface_level=request.render_options.surface_level,
            sampling_grid=sampling_grid,
        )
    except Exception as error:
        raise _output_error(request, "camera_view_surface", error) from error

    render_size = request.comparison_options.resolve_render_size(
        request.render_options.image_size
    )
    rows = []
    raw_stack = []
    aligned_stack = []
    exact_stack = []
    exact_search_stack = []
    exact_matched_stack = []
    near_stack = []
    near_search_stack = []
    near_matched_stack = []
    panel_rows = []
    family_ranks = {name: 0 for name in request.search_result.families}

    for row_number, candidate in enumerate(request.search_result.rows, start=1):
        try:
            refined = (
                None
                if request.refinement is None
                else request.refinement.rows[row_number - 1]
            )
        except Exception as error:
            raise _candidate_output_error(
                candidate,
                "near-axis-result",
                error,
            ) from error
        family_ranks[candidate.family_name] += 1
        rank = family_ranks[candidate.family_name]
        display_roll = float(request.axis_rolls.get(candidate.family_name, 0.0))
        raw_stack.append(candidate.raw_class)
        try:
            aligned_stack.append(
                apply_axis_display_roll(
                    candidate.aligned_class,
                    display_roll,
                    background=request.render_options.background,
                )
            )
        except Exception as error:
            raise _candidate_output_error(
                candidate,
                "axis_candidates_aligned",
                error,
            ) from error
        native_exact = _render_native_match(
            request,
            candidate,
            candidate.exact_rotation_matrix,
            reference_rotation_matrix=candidate.canonical_axis_rotation_matrix,
            pass_name="exact",
        )
        try:
            exact_stack.append(
                apply_axis_display_roll(
                    np.flipud(native_exact["reference_projection"]),
                    display_roll,
                    background=request.render_options.background,
                )
            )
        except Exception as error:
            raise _candidate_output_error(
                candidate,
                "axis_exact_references",
                error,
            ) from error
        exact_search_stack.append(candidate.search_projection)
        exact_matched_stack.append(native_exact["matched_projection"])
        exact_view = _render_camera_view(
            request,
            directory / "renders" / "exact",
            surface,
            candidate,
            candidate.exact_rotation_matrix,
            row_number,
            render_size,
            pass_name="exact",
        )
        native_near = None
        auto_crop_decision = None
        if refined is None:
            auto_crop_decision = _compute_axis_auto_crop_decision(
                request,
                warnings,
                surface=surface,
                candidate=candidate,
                native_exact=native_exact,
                native_near=None,
                display_roll=display_roll,
            )
            try:
                panel_rows.append(
                    ExactAxisResultPanelRow(
                        label=AxisResultLabel(
                            family_name=candidate.family_name,
                            rank=rank,
                            class_number=candidate.class_number,
                            axis_class_score=candidate.exact_score,
                        ),
                        class_average=np.flipud(candidate.raw_class),
                        exact_matched_projection=np.flipud(
                            native_exact["matched_projection"]
                        ),
                        exact_axis_view=exact_view,
                        auto_crop_decision=auto_crop_decision,
                    )
                )
            except Exception as error:
                raise _candidate_output_error(
                    candidate,
                    "axis_search_preview",
                    error,
                ) from error
        else:
            native_near = _render_native_match(
                request,
                candidate,
                refined.near_axis_rotation_matrix,
                reference_rotation_matrix=(
                    refined.canonical_near_axis_rotation_matrix
                ),
                pass_name="near",
            )
            try:
                near_stack.append(
                    apply_axis_display_roll(
                        np.flipud(native_near["reference_projection"]),
                        display_roll,
                        background=request.render_options.background,
                    )
                )
            except Exception as error:
                raise _candidate_output_error(
                    candidate,
                    "axis_near_projections",
                    error,
                ) from error
            near_search_stack.append(refined.matched_search_projection)
            near_matched_stack.append(native_near["matched_projection"])
            near_view = _render_camera_view(
                request,
                directory / "renders" / "near",
                surface,
                candidate,
                refined.near_axis_rotation_matrix,
                row_number,
                render_size,
                pass_name="near",
            )
            auto_crop_decision = _compute_axis_auto_crop_decision(
                request,
                warnings,
                surface=surface,
                candidate=candidate,
                native_exact=native_exact,
                native_near=native_near,
                near_rotation_matrix=refined.near_axis_rotation_matrix,
                display_roll=display_roll,
            )
            try:
                panel_rows.append(
                    AxisResultPanelRow(
                        label=AxisResultLabel(
                            family_name=candidate.family_name,
                            rank=rank,
                            class_number=candidate.class_number,
                            axis_class_score=candidate.exact_score,
                            near_axis_score=refined.refined_score,
                            near_axis_angle_degrees=refined.angular_distance_degrees,
                        ),
                        axis_aligned_class=np.flipud(candidate.raw_class),
                        near_axis_projection=np.flipud(
                            native_near["matched_projection"]
                        ),
                        exact_axis_projection=np.flipud(
                            native_exact["matched_projection"]
                        ),
                        near_axis_view=near_view,
                        exact_axis_view=exact_view,
                        auto_crop_decision=auto_crop_decision,
                    )
                )
            except Exception as error:
                raise _candidate_output_error(
                    candidate,
                    "axis_search_preview",
                    error,
                ) from error
        try:
            row = _result_row(
                request,
                candidate,
                refined,
                native_exact,
                native_near,
                rank,
                display_roll,
            )
        except Exception as error:
            raise _candidate_output_error(
                candidate,
                "axis_search_results.json",
                error,
            ) from error
        rows.append(row)
        if auto_crop_decision is not None:
            row["presentation"] = {
                "auto_crop_2d": auto_crop_decision.as_dict()
            }
        _emit(
            request,
            AxisResultRenderingEventCode.CANDIDATE_COMPLETED,
            "progress",
            "result-rendering",
            "candidate completed",
            family_name=candidate.family_name,
            class_number=candidate.class_number,
        )

    timings["result-rendering"] = {
        "elapsed_seconds": request.clock() - render_started_at
    }
    output_started_at = request.clock()
    _emit(
        request,
        AxisResultRenderingEventCode.OUTPUT_WRITING_STARTED,
        "progress",
        "output-writing",
        "started",
    )
    stack_values = {
        "axis_candidates_raw": np.asarray(raw_stack, dtype=np.float32),
        "axis_candidates_aligned": np.asarray(aligned_stack, dtype=np.float32),
        "axis_exact_references": np.asarray(exact_stack, dtype=np.float32),
        "axis_exact_search_projections": np.asarray(
            exact_search_stack, dtype=np.float32
        ),
        "axis_exact_matched_projections": np.asarray(
            exact_matched_stack, dtype=np.float32
        ),
        "axis_search_preview": np.asarray(aligned_stack, dtype=np.float32),
    }
    if request.refine_near_axis:
        stack_values.update(
            {
                "axis_near_projections": np.asarray(near_stack, dtype=np.float32),
                "axis_near_search_projections": np.asarray(
                    near_search_stack, dtype=np.float32
                ),
                "axis_near_matched_projections": np.asarray(
                    near_matched_stack, dtype=np.float32
                ),
            }
        )
    stacks = {}
    for name, stack in stack_values.items():
        pixel_size_A = (
            float(
                request.search_result.rows[0].score_metadata["search_pixel_size_A"]
            )
            if "search_projections" in name
            else request.class_pixel_size_A
        )
        filename = f"{name}.mrcs"
        try:
            mrc.write(directory / filename, stack, pixel_size_A)
        except Exception as error:
            raise _output_error(request, name, error) from error
        stacks[name] = AxisRenderedStack(filename, stack, pixel_size_A)

    preview_pages = []
    for start in range(0, len(panel_rows), request.comparison_options.page_size):
        page_number = len(preview_pages) + 1
        output_name = f"axis_search_preview_{page_number:03d}.png"
        try:
            page = create_axis_result_figure(
                panel_rows[start : start + request.comparison_options.page_size],
                axis_rolls=request.axis_rolls,
                dpi=request.comparison_options.dpi,
                background=request.render_options.background,
                comparison_options=request.comparison_options,
            )
            page.savefig(
                directory / output_name,
                dpi=request.comparison_options.dpi,
            )
        except Exception as error:
            raise _output_error(request, output_name, error) from error
        preview_pages.append(page)

    timings["output-writing"] = {
        "elapsed_seconds": request.clock() - output_started_at
    }
    if request.run_started_at is not None:
        timings["total"] = {
            "elapsed_seconds": request.clock() - request.run_started_at
        }
    output_names = list(stack_values)
    artifact = _artifact(
        request,
        rows,
        output_names,
        sampling_grid,
        surface,
        render_size,
        timings,
    )
    try:
        (directory / "axis_search_results.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
    except Exception as error:
        raise _output_error(request, "axis_search_results.json", error) from error
    return AxisResultRenderingResult(
        artifact=artifact,
        stacks=stacks,
        preview_pages=tuple(preview_pages),
        preview_path=directory / "axis_search_preview_001.png",
    )


def _render_native_match(
    request,
    candidate,
    rotation_matrix,
    *,
    reference_rotation_matrix,
    pass_name,
):
    try:
        native = project_native_matched_projection(
            candidate.raw_class,
            request.matching_map,
            rotation_matrix,
            class_pixel_size=request.class_pixel_size_A,
            volume_pixel_size=request.map_pixel_size_A,
        )
    except Exception as error:
        output_name = f"axis_{pass_name}_matched_projections"
        raise _candidate_output_error(candidate, output_name, error) from error
    try:
        grid = prepare_native_matching_grid(
            candidate.raw_class,
            request.matching_map,
            class_pixel_size=request.class_pixel_size_A,
            volume_pixel_size=request.map_pixel_size_A,
        )
        reference_projection = project_volume_at_rotation(
            grid.volume,
            reference_rotation_matrix,
        )
    except Exception as error:
        output_name = (
            "axis_exact_references"
            if pass_name == "exact"
            else "axis_near_projections"
        )
        raise _candidate_output_error(candidate, output_name, error) from error
    try:
        diagnostic = compute_diagnostic_band_limited_score(
            candidate.raw_class,
            native.matched_projection,
            pixel_size_A=native.projection_pixel_size_A,
            settings=request.config.score_config(),
        )
    except Exception as error:
        raise _candidate_output_error(
            candidate,
            "axis_search_results.json",
            error,
        ) from error
    return {
        "projection": native.projection,
        "reference_projection": reference_projection.astype(np.float32, copy=False),
        "matched_projection": native.matched_projection,
        "shift_xy_pixels": [
            float(native.projection_shift_pixels[0]),
            float(native.projection_shift_pixels[1]),
        ],
        "pixel_size_A": float(native.projection_pixel_size_A),
        "diagnostic_score": {
            "score": diagnostic.score,
            "valid": diagnostic.valid,
            "invalid_reason": diagnostic.invalid_reason,
            **diagnostic.metadata,
        },
    }


def _render_camera_view(
    request,
    directory,
    surface,
    candidate,
    rotation_matrix,
    row_number,
    render_size,
    *,
    pass_name,
):
    try:
        path = write_camera_view_render(
            directory,
            surface=surface,
            rotation_matrix=rotation_matrix,
            class_number=row_number,
            image_size=render_size,
            background=request.render_options.background,
        )
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")).copy()
    except Exception as error:
        output_name = f"renders/{pass_name}/class_{row_number:03d}_exact.png"
        raise _candidate_output_error(candidate, output_name, error) from error


def _compute_axis_auto_crop_decision(
    request,
    warnings,
    *,
    surface,
    candidate,
    native_exact,
    native_near,
    near_rotation_matrix=None,
    display_roll=0.0,
):
    if not request.comparison_options.auto_crop_2d:
        return None
    projections = [
        _roll_projection_for_auto_crop(
            native_exact["matched_projection"], display_roll
        )
    ]
    rotations = [candidate.exact_rotation_matrix]
    if native_near is not None:
        projections.append(
            _roll_projection_for_auto_crop(
                native_near["matched_projection"], display_roll
            )
        )
        rotations.append(near_rotation_matrix)
    try:
        silhouettes = [
            get_surface_silhouette_bounds(
                surface,
                rotation,
                display_roll_degrees=display_roll,
            )
            for rotation in rotations
        ]
    except (TypeError, ValueError):
        silhouettes = []
    decision = compute_auto_crop_2d_framing(
        projections,
        silhouettes,
        enabled=True,
    )
    if decision.fallback:
        event = AxisResultRenderingEvent(
            AxisResultRenderingEventCode.WARNING,
            "warning",
            "result-rendering",
            "Auto-Cropped 2D Framing fell back for "
            f"Class {candidate.class_number}: {decision.fallback_reason}",
            family_name=candidate.family_name,
            class_number=candidate.class_number,
            output_name="axis_search_preview",
        )
        warnings.append(event)
        _notify(request.warning_callback, event)
    return decision


def _roll_projection_for_auto_crop(projection, display_roll):
    displayed = np.flipud(np.asarray(projection))
    border = np.concatenate(
        [
            displayed[0, :],
            displayed[-1, :],
            displayed[1:-1, 0],
            displayed[1:-1, -1],
        ]
    )
    background = float(np.median(border))
    return (
        apply_axis_display_roll(
            displayed - background,
            display_roll,
            background="dark",
        )
        + background
    )


def _result_row(
    request,
    candidate,
    refined,
    native_exact,
    native_near,
    rank,
    display_roll,
):
    return {
        "family": candidate.family_name,
        "rank": rank,
        "class_number": candidate.class_number,
        "axis_class_score": candidate.exact_score,
        "raw_correlation": candidate.raw_correlation,
        "mirrored_score": candidate.mirrored_score,
        "roll_degrees": candidate.roll_degrees,
        "shift_xy_pixels": list(candidate.shift_xy_pixels),
        "search_shift_xy_pixels": list(candidate.shift_xy_pixels),
        "search_box_size": int(candidate.score_metadata["search_box_size"]),
        "search_pixel_size_A": float(
            candidate.score_metadata["search_pixel_size_A"]
        ),
        "search_evaluation_count": int(
            candidate.score_metadata["search_evaluation_count"]
        ),
        "native_box_size": int(candidate.raw_class.shape[0]),
        "native_pixel_size_A": float(request.class_pixel_size_A),
        "native_shift_xy_pixels": native_exact["shift_xy_pixels"],
        "score_provenance": candidate.score_metadata,
        "native_exact_diagnostic_score": native_exact["diagnostic_score"],
        "refined_score": None if refined is None else refined.refined_score,
        "angular_distance_degrees": (
            None if refined is None else refined.angular_distance_degrees
        ),
        "exact_axis_rotation_matrix": candidate.exact_rotation_matrix.tolist(),
        "canonical_exact_axis_rotation_matrix": (
            candidate.canonical_axis_rotation_matrix.tolist()
        ),
        "near_axis_rotation_matrix": (
            None if refined is None else refined.near_axis_rotation_matrix.tolist()
        ),
        "canonical_near_axis_rotation_matrix": (
            None
            if refined is None
            else refined.canonical_near_axis_rotation_matrix.tolist()
        ),
        "native_near_diagnostic_score": (
            None if native_near is None else native_near["diagnostic_score"]
        ),
        "axis_roll_degrees": display_roll,
        "cone_boundary": None if refined is None else refined.cone_boundary,
        "duplicate": candidate.duplicate,
        "warnings": list(
            candidate.warnings + (() if refined is None else refined.warnings)
        ),
    }


def _artifact(
    request,
    rows,
    output_names,
    sampling_grid,
    surface,
    render_size,
    timings,
):
    artifact = {
        "cryosparc_version": "5.0.6",
        "symmetry": "I",
        "families": list(request.search_result.families),
        "family_diagnostics": {
            name: {
                "first_score": ranking.first_score,
                "second_score": ranking.second_score,
                "score_margin": ranking.score_margin,
            }
            for name, ranking in request.search_result.families.items()
        },
        "config": _config_metadata(request.config),
        "proximity_config": {
            "enabled": bool(request.refine_near_axis),
            "cone_degrees": request.proximity_config.cone_degrees,
            "coarse_step_degrees": request.proximity_config.coarse_step_degrees,
            "refine_step_degrees": request.proximity_config.refine_step_degrees,
        },
        "presentation": {
            "columns": list(
                AXIS_RESULT_COLUMNS
                if request.refine_near_axis
                else EXACT_AXIS_RESULT_COLUMNS
            ),
            "axis_rolls": request.axis_rolls,
            "comparison_dpi": request.comparison_options.dpi,
            "preview_page_size": request.comparison_options.page_size,
            "rendering_map": request.render_options.map_name,
            "render_size": render_size,
            "render_grid_size": request.render_options.grid_size,
            "surface_sampling": sampling_grid.as_dict(),
            "surface_level": surface.surface_level,
            "background": request.render_options.background,
            "static_only": True,
        },
        "outputs": output_names,
        "timings": timings,
        "rows": rows,
    }
    if request.comparison_options.auto_crop_2d:
        artifact["presentation"]["auto_crop_2d"] = {
            "enabled": True,
            "max_zoom": AUTO_CROP_MAX_ZOOM,
            "padding_fraction": AUTO_CROP_PADDING_FRACTION,
        }
    return artifact


def _sampling_grid_message(sampling_grid):
    original_shape = " x ".join(map(str, sampling_grid.original_shape))
    sampled_shape = " x ".join(map(str, sampling_grid.sampled_shape))
    requested_grid = (
        "native"
        if sampling_grid.requested_grid_size is None
        else str(sampling_grid.requested_grid_size)
    )
    return (
        "Surface Sampling Grid: "
        f"original={original_shape}; requested={requested_grid}; "
        f"effective={sampled_shape}; mode={sampling_grid.mode}; "
        f"downsampled={'yes' if sampling_grid.was_downsampled else 'no'}; "
        "estimated minimum working memory="
        f"{sampling_grid.estimated_memory_gib:.3f} GiB "
        "(mesh and plotting allocations excluded)."
    )


def _config_metadata(config):
    return {
        "search_max_size": config.search_max_size,
        "low_resolution_A": config.low_resolution_A,
        "high_resolution_A": config.high_resolution_A,
        "mask_radius_fraction": config.mask_radius_fraction,
        "mask_edge_fraction": config.mask_edge_fraction,
        "roll_coarse_step_degrees": config.roll_coarse_step_degrees,
        "roll_refine_step_degrees": config.roll_refine_step_degrees,
        "shift_bound_fraction": config.shift_bound_fraction,
        "top_n": config.top_n,
        "mirror_warning_margin": config.mirror_warning_margin,
    }


def _emit(
    request,
    code,
    kind,
    stage,
    message,
    *,
    family_name=None,
    class_number=None,
    output_name=None,
):
    callback = (
        request.warning_callback if kind == "warning" else request.progress_callback
    )
    _notify(
        callback,
        AxisResultRenderingEvent(
            code,
            kind,
            stage,
            message,
            family_name=family_name,
            class_number=class_number,
            output_name=output_name,
        ),
    )


def _notify(callback, event):
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        pass


def _output_error(request, output_name, error):
    candidates = "; ".join(
        f"family={candidate.family_name} class={candidate.class_number}"
        for candidate in request.search_result.rows
    )
    return AxisResultRenderingError(
        f"Result Rendering {candidates} output={output_name} failed: "
        f"{type(error).__name__}: {error}"
    )


def _candidate_output_error(candidate, output_name, error):
    return AxisResultRenderingError(
        "Result Rendering "
        f"family={candidate.family_name} class={candidate.class_number} "
        f"output={output_name} failed: {type(error).__name__}: {error}"
    )


def _promote_complete_result(staging_directory, output_directory, request):
    entries = tuple(staging_directory.iterdir())
    managed_targets = [
        output_directory / name
        for name in _MANAGED_RESULT_NAMES
        if (output_directory / name).exists()
    ]
    managed_targets.extend(output_directory.glob("axis_search_preview_*.png"))
    with tempfile.TemporaryDirectory(
        dir=output_directory,
        prefix=".axis-result-backup-",
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
            raise _output_error(request, "local-result-set", error) from error
