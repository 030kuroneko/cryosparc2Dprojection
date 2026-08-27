"""Supported CryoSPARC External Job boundary for Symmetry-Axis Search."""

from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic

import numpy as np
from cryosparc import mrc
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage import shift as shift_image

from cryosparc_2d_projection.axis_search import (
    AxisProximityConfig,
    AxisSearchConfig,
    rank_axis_families,
    refine_axis_candidates,
)
from cryosparc_2d_projection.axis_presentation import (
    AXIS_RESULT_COLUMNS,
    EXACT_AXIS_RESULT_COLUMNS,
    AxisResultPanelRow,
    ExactAxisResultPanelRow,
    apply_axis_display_roll,
    create_axis_result_figure,
)
from cryosparc_2d_projection.matching_grid import prepare_native_matching_grid
from cryosparc_2d_projection.projection import (
    find_projection_shift,
    project_volume_at_rotation,
)
from cryosparc_2d_projection.scoring import compute_diagnostic_band_limited_score
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    build_surface_model,
    resolve_surface_sampling_grid,
    write_camera_view_render,
)


@dataclass(frozen=True)
class AxisSourceOutput:
    job_uid: str
    output_name: str


def run_axis_search_job(
    project,
    workspace_uid,
    templates_source,
    volume_source,
    *,
    families=None,
    config=None,
    proximity_config=None,
    axis_rolls=None,
    comparison_options=None,
    render_options=None,
    refine_near_axis=False,
    status_callback=None,
    warning_callback=None,
    progress_clock=monotonic,
    heartbeat_seconds=30.0,
    stalled_warning_seconds=300.0,
):
    """Create and execute an image-only Axis Search External Job."""

    config = config or AxisSearchConfig()
    proximity_config = proximity_config or AxisProximityConfig()
    axis_rolls = dict(axis_rolls or {})
    comparison_options = comparison_options or ComparisonRenderOptions()
    render_options = render_options or ClassRenderOptions()
    job = project.create_external_job(
        workspace_uid,
        title="Symmetry-Axis Class Search (CryoSPARC 5.0.6)",
    )
    _add_input(
        job,
        name="templates",
        type="template",
        slots=["blob"],
        source=templates_source,
        title="Selected 2D class averages",
    )
    _add_input(
        job,
        name="volume",
        type="volume",
        slots=(
            ["map", "map_sharp"]
            if render_options.map_name == "sharpened"
            else ["map"]
        ),
        source=volume_source,
        title="Unsharpened Matching Map",
    )
    output_names = [
        "axis_candidates_raw",
        "axis_candidates_aligned",
        "axis_exact_references",
        "axis_exact_search_projections",
        "axis_exact_matched_projections",
        "axis_search_preview",
    ]
    if refine_near_axis:
        output_names.extend(
            [
                "axis_near_projections",
                "axis_near_search_projections",
                "axis_near_matched_projections",
            ]
        )
    for name in output_names:
        job.add_output(
            type="template",
            name=name,
            slots=["blob"],
            title=name.replace("_", " ").title(),
        )

    with job.run():
        run_started_at = monotonic()
        timings = {}
        _safe_status(
            job,
            "Axis Search stage: stage=input-loading status=started",
            status_callback,
        )
        stage_started_at = monotonic()
        classes, class_pixel_size_A = _load_templates(
            project, job.load_input("templates")
        )
        volume_input = job.load_input("volume")
        volume_path = _resolve_project_path(project, volume_input["map/path"][0])
        _, matching_map = mrc.read(volume_path)
        map_pixel_size_A = float(volume_input["map/psize_A"][0])
        rendering_slot = (
            "map_sharp" if render_options.map_name == "sharpened" else "map"
        )
        rendering_path = _resolve_project_path(
            project, volume_input[f"{rendering_slot}/path"][0]
        )
        _, rendering_map = mrc.read(rendering_path)
        timings["input-loading"] = {
            "elapsed_seconds": monotonic() - stage_started_at
        }
        _safe_status(
            job,
            "Axis Search stage: stage=input-loading status=completed "
            f"elapsed={timings['input-loading']['elapsed_seconds']:.3f}s",
            status_callback,
        )
        _safe_status(
            job,
            "Axis Search stage: stage=exact-ranking status=started "
            f"classes={len(classes)} search_max_size={config.search_max_size}",
            status_callback,
        )
        stage_started_at = monotonic()
        progress_reporter = _AxisProgressReporter(
            job,
            status_callback=status_callback,
            warning_callback=warning_callback,
            clock=progress_clock,
            heartbeat_seconds=heartbeat_seconds,
            stalled_seconds=stalled_warning_seconds,
        )
        try:
            search_result = rank_axis_families(
                classes,
                matching_map,
                families=families,
                class_pixel_size_A=class_pixel_size_A,
                map_pixel_size_A=map_pixel_size_A,
                config=config,
                progress_callback=progress_reporter,
            )
        except Exception as error:
            _safe_status(
                job,
                "Axis Search stage: stage=exact-ranking status=failed "
                f"{progress_reporter.context()} "
                f"elapsed={monotonic() - stage_started_at:.3f}s "
                f"error={type(error).__name__}: {error}",
                status_callback,
            )
            raise
        timings["exact-ranking"] = {
            "elapsed_seconds": monotonic() - stage_started_at
        }
        _safe_status(
            job,
            "Axis Search stage: stage=exact-ranking status=completed "
            f"elapsed={timings['exact-ranking']['elapsed_seconds']:.3f}s",
            status_callback,
        )
        refinement = None
        if refine_near_axis:
            _safe_status(
                job,
                "Axis Search stage: stage=near-axis-refinement status=started",
                status_callback,
            )
            stage_started_at = monotonic()
            try:
                refinement = refine_axis_candidates(
                    search_result,
                    matching_map,
                    class_pixel_size_A=class_pixel_size_A,
                    map_pixel_size_A=map_pixel_size_A,
                    config=proximity_config,
                    progress_callback=progress_reporter,
                )
            except Exception as error:
                _safe_status(
                    job,
                    "Axis Search stage: stage=near-axis-refinement status=failed "
                    f"{progress_reporter.context()} "
                    f"elapsed={monotonic() - stage_started_at:.3f}s "
                    f"error={type(error).__name__}: {error}",
                    status_callback,
                )
                raise
            timings["near-axis-refinement"] = {
                "elapsed_seconds": monotonic() - stage_started_at
            }
            _safe_status(
                job,
                "Axis Search stage: stage=near-axis-refinement status=completed "
                f"elapsed={timings['near-axis-refinement']['elapsed_seconds']:.3f}s",
                status_callback,
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
        _safe_status(
            job,
            "Axis Search stage: stage=result-rendering status=started",
            status_callback,
        )
        stage_started_at = monotonic()
        sampling_grid = resolve_surface_sampling_grid(
            rendering_map.shape, render_options.grid_size
        )
        original_shape = " x ".join(map(str, sampling_grid.original_shape))
        sampled_shape = " x ".join(map(str, sampling_grid.sampled_shape))
        requested_grid = (
            "native"
            if sampling_grid.requested_grid_size is None
            else str(sampling_grid.requested_grid_size)
        )
        _safe_status(
            job,
            "Surface Sampling Grid: "
            f"original={original_shape}; requested={requested_grid}; "
            f"effective={sampled_shape}; mode={sampling_grid.mode}; "
            f"downsampled={'yes' if sampling_grid.was_downsampled else 'no'}; "
            "estimated minimum working memory="
            f"{sampling_grid.estimated_memory_gib:.3f} GiB "
            "(mesh and plotting allocations excluded).",
            status_callback,
        )
        for warning in sampling_grid.warnings:
            _safe_warning(job, warning, warning_callback)
        try:
            surface = build_surface_model(
                rendering_map,
                surface_level=render_options.surface_level,
                sampling_grid=sampling_grid,
            )
        except Exception as error:
            _safe_status(
                job,
                "Axis Search stage: stage=result-rendering status=failed "
                f"elapsed={monotonic() - stage_started_at:.3f}s "
                f"error={type(error).__name__}: {error}",
                status_callback,
            )
            raise
        render_size = comparison_options.resolve_render_size(
            render_options.image_size
        )
        family_ranks = {name: 0 for name in search_result.families}
        for row_number, candidate in enumerate(search_result.rows, start=1):
            refined = (
                None if refinement is None else refinement.rows[row_number - 1]
            )
            family_ranks[candidate.family_name] += 1
            rank = family_ranks[candidate.family_name]
            display_roll = float(axis_rolls.get(candidate.family_name, 0.0))
            raw_stack.append(candidate.raw_class)
            aligned_stack.append(
                apply_axis_display_roll(
                    candidate.aligned_class,
                    display_roll,
                    background=render_options.background,
                )
            )
            try:
                native_exact = _native_axis_match(
                    candidate.raw_class,
                    matching_map,
                    candidate.exact_rotation_matrix,
                    reference_rotation_matrix=(
                        candidate.canonical_axis_rotation_matrix
                    ),
                    roll_degrees=0.0,
                    class_pixel_size_A=class_pixel_size_A,
                    map_pixel_size_A=map_pixel_size_A,
                    score_config=config.score_config(),
                )
            except Exception as error:
                _safe_status(
                    job,
                    "Axis Search stage: stage=native-reprojection status=failed "
                    f"family={candidate.family_name} "
                    f"class={candidate.class_number} pass=exact "
                    f"error={type(error).__name__}: {error}",
                    status_callback,
                )
                raise
            exact_stack.append(
                apply_axis_display_roll(
                    np.flipud(native_exact["reference_projection"]),
                    display_roll,
                    background=render_options.background,
                )
            )
            exact_search_stack.append(candidate.search_projection)
            exact_matched_stack.append(native_exact["matched_projection"])
            exact_render_path = write_camera_view_render(
                _resource_directory(job) / "renders" / "exact",
                surface=surface,
                rotation_matrix=candidate.exact_rotation_matrix,
                class_number=row_number,
                image_size=render_size,
                background=render_options.background,
            )
            from PIL import Image

            with Image.open(exact_render_path) as image:
                exact_view = np.asarray(image.convert("RGB")).copy()
            native_near = None
            if refined is None:
                panel_rows.append(
                    ExactAxisResultPanelRow(
                        family_name=candidate.family_name,
                        rank=rank,
                        class_number=candidate.class_number,
                        class_average=np.flipud(candidate.raw_class),
                        exact_matched_projection=np.flipud(
                            native_exact["matched_projection"]
                        ),
                        exact_axis_view=exact_view,
                        axis_class_score=candidate.exact_score,
                    )
                )
            else:
                try:
                    native_near = _native_axis_match(
                        candidate.raw_class,
                        matching_map,
                        refined.near_axis_rotation_matrix,
                        reference_rotation_matrix=(
                            refined.canonical_near_axis_rotation_matrix
                        ),
                        roll_degrees=0.0,
                        class_pixel_size_A=class_pixel_size_A,
                        map_pixel_size_A=map_pixel_size_A,
                        score_config=config.score_config(),
                    )
                except Exception as error:
                    _safe_status(
                        job,
                        "Axis Search stage: stage=native-reprojection "
                        "status=failed "
                        f"family={candidate.family_name} "
                        f"class={candidate.class_number} pass=near "
                        f"error={type(error).__name__}: {error}",
                        status_callback,
                    )
                    raise
                near_stack.append(
                    apply_axis_display_roll(
                        np.flipud(native_near["reference_projection"]),
                        display_roll,
                        background=render_options.background,
                    )
                )
                near_search_stack.append(refined.matched_search_projection)
                near_matched_stack.append(native_near["matched_projection"])
                near_render_path = write_camera_view_render(
                    _resource_directory(job) / "renders" / "near",
                    surface=surface,
                    rotation_matrix=refined.near_axis_rotation_matrix,
                    class_number=row_number,
                    image_size=render_size,
                    background=render_options.background,
                )
                with Image.open(near_render_path) as image:
                    near_view = np.asarray(image.convert("RGB")).copy()
                panel_rows.append(
                    AxisResultPanelRow(
                        family_name=candidate.family_name,
                        rank=rank,
                        class_number=candidate.class_number,
                        axis_aligned_class=np.flipud(candidate.raw_class),
                        near_axis_projection=np.flipud(
                            native_near["matched_projection"]
                        ),
                        exact_axis_projection=np.flipud(
                            native_exact["matched_projection"]
                        ),
                        near_axis_view=near_view,
                        exact_axis_view=exact_view,
                        axis_class_score=candidate.exact_score,
                        near_axis_score=refined.refined_score,
                    )
                )
            rows.append(
                {
                    "family": candidate.family_name,
                    "rank": rank,
                    "class_number": candidate.class_number,
                    "axis_class_score": candidate.exact_score,
                    "raw_correlation": candidate.raw_correlation,
                    "mirrored_score": candidate.mirrored_score,
                    "roll_degrees": candidate.roll_degrees,
                    "shift_xy_pixels": list(candidate.shift_xy_pixels),
                    "search_shift_xy_pixels": list(candidate.shift_xy_pixels),
                    "search_box_size": int(
                        candidate.score_metadata["search_box_size"]
                    ),
                    "search_pixel_size_A": float(
                        candidate.score_metadata["search_pixel_size_A"]
                    ),
                    "search_evaluation_count": int(
                        candidate.score_metadata["search_evaluation_count"]
                    ),
                    "native_box_size": int(candidate.raw_class.shape[0]),
                    "native_pixel_size_A": float(class_pixel_size_A),
                    "native_shift_xy_pixels": native_exact["shift_xy_pixels"],
                    "score_provenance": candidate.score_metadata,
                    "native_exact_diagnostic_score": native_exact[
                        "diagnostic_score"
                    ],
                    "refined_score": (
                        None if refined is None else refined.refined_score
                    ),
                    "angular_distance_degrees": (
                        None if refined is None else refined.angular_distance_degrees
                    ),
                    "exact_axis_rotation_matrix": candidate.exact_rotation_matrix.tolist(),
                    "canonical_exact_axis_rotation_matrix": (
                        candidate.canonical_axis_rotation_matrix.tolist()
                    ),
                    "near_axis_rotation_matrix": (
                        None
                        if refined is None
                        else refined.near_axis_rotation_matrix.tolist()
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
                    "cone_boundary": (
                        None if refined is None else refined.cone_boundary
                    ),
                    "duplicate": candidate.duplicate,
                    "warnings": list(
                        candidate.warnings
                        + (() if refined is None else refined.warnings)
                    ),
                }
            )
        timings["result-rendering"] = {
            "elapsed_seconds": monotonic() - stage_started_at
        }
        _safe_status(
            job,
            "Axis Search stage: stage=result-rendering status=completed "
            f"elapsed={timings['result-rendering']['elapsed_seconds']:.3f}s",
            status_callback,
        )
        _safe_status(
            job,
            "Axis Search stage: stage=output-writing status=started",
            status_callback,
        )
        stage_started_at = monotonic()
        stacks = {
            "axis_candidates_raw": np.asarray(raw_stack, dtype=np.float32),
            "axis_candidates_aligned": np.asarray(aligned_stack, dtype=np.float32),
            "axis_exact_references": np.asarray(exact_stack, dtype=np.float32),
            "axis_exact_search_projections": np.asarray(
                exact_search_stack, dtype=np.float32
            ),
            "axis_exact_matched_projections": np.asarray(
                exact_matched_stack, dtype=np.float32
            ),
            "axis_search_preview": np.asarray(
                aligned_stack, dtype=np.float32
            ),
        }
        if refine_near_axis:
            stacks.update(
                {
                    "axis_near_projections": np.asarray(
                        near_stack, dtype=np.float32
                    ),
                    "axis_near_search_projections": np.asarray(
                        near_search_stack, dtype=np.float32
                    ),
                    "axis_near_matched_projections": np.asarray(
                        near_matched_stack, dtype=np.float32
                    ),
                }
            )
        job_directory = _resource_directory(job)
        for name, stack in stacks.items():
            filename = f"{name}.mrcs"
            pixel_size_A = (
                float(search_result.rows[0].score_metadata["search_pixel_size_A"])
                if "search_projections" in name
                else class_pixel_size_A
            )
            mrc.write(job_directory / filename, stack, pixel_size_A)
            _save_template_stack(
                job,
                name,
                filename,
                stack,
                pixel_size_A=pixel_size_A,
            )
        artifact = {
            "cryosparc_version": "5.0.6",
            "symmetry": "I",
            "families": list(search_result.families),
            "family_diagnostics": {
                name: {
                    "first_score": ranking.first_score,
                    "second_score": ranking.second_score,
                    "score_margin": ranking.score_margin,
                }
                for name, ranking in search_result.families.items()
            },
            "config": _config_metadata(config),
            "proximity_config": {
                "enabled": bool(refine_near_axis),
                "cone_degrees": proximity_config.cone_degrees,
                "coarse_step_degrees": proximity_config.coarse_step_degrees,
                "refine_step_degrees": proximity_config.refine_step_degrees,
            },
            "presentation": {
                "columns": list(
                    AXIS_RESULT_COLUMNS
                    if refine_near_axis
                    else EXACT_AXIS_RESULT_COLUMNS
                ),
                "axis_rolls": axis_rolls,
                "comparison_dpi": comparison_options.dpi,
                "preview_page_size": comparison_options.page_size,
                "rendering_map": render_options.map_name,
                "render_size": render_size,
                "render_grid_size": render_options.grid_size,
                "surface_sampling": sampling_grid.as_dict(),
                "surface_level": surface.surface_level,
                "background": render_options.background,
                "static_only": True,
            },
            "outputs": list(output_names),
            "timings": timings,
            "rows": rows,
        }
        preview_pages = []
        for start in range(0, len(panel_rows), comparison_options.page_size):
            page = create_axis_result_figure(
                panel_rows[start : start + comparison_options.page_size],
                axis_rolls=axis_rolls,
                dpi=comparison_options.dpi,
                background=render_options.background,
            )
            page_number = len(preview_pages) + 1
            page.savefig(
                job_directory / f"axis_search_preview_{page_number:03d}.png",
                dpi=comparison_options.dpi,
            )
            preview_pages.append(page)
        for page_number, page in enumerate(preview_pages, start=1):
            job.log_plot(
                page,
                f"Symmetry-Axis Search preview {page_number}/{len(preview_pages)}",
                formats=["png"],
                savefig_kw={"dpi": comparison_options.dpi, "bbox_inches": "tight"},
            )
        _attach_axis_dashboard_preview(
            job,
            job_directory / "axis_search_preview_001.png",
            status_callback=status_callback,
        )
        for row in rows:
            refined_text = (
                "disabled"
                if row["refined_score"] is None
                else f"{row['refined_score']:.6f}"
            )
            angular_text = (
                "disabled"
                if row["angular_distance_degrees"] is None
                else f"{row['angular_distance_degrees']:.3f}"
            )
            _safe_status(
                job,
                "Axis Search row: "
                f"family={row['family']} rank={row['rank']} "
                f"class={row['class_number']} "
                f"exact_score={row['axis_class_score']:.6f} "
                f"refined_score={refined_text} "
                f"angular_distance={angular_text} "
                f"duplicate={row['duplicate']} warnings={row['warnings']}",
                status_callback,
            )
            _safe_job_log(
                job,
                "Axis Search row JSON: "
                + json.dumps(row, sort_keys=True, separators=(",", ":")),
            )
        _safe_status(
            job,
            f"Ranked {len(classes)} classes across "
            f"{len(search_result.families)} Axis Families.",
            status_callback,
        )
        timings["output-writing"] = {
            "elapsed_seconds": monotonic() - stage_started_at
        }
        timings["total"] = {"elapsed_seconds": monotonic() - run_started_at}
        (job_directory / "axis_search_results.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
        _safe_status(
            job,
            "Axis Search stage: stage=output-writing status=completed "
            f"elapsed={timings['output-writing']['elapsed_seconds']:.3f}s",
            status_callback,
        )
    return artifact


class _AxisProgressReporter:
    def __init__(
        self,
        job,
        *,
        status_callback=None,
        warning_callback=None,
        clock=monotonic,
        heartbeat_seconds=30.0,
        stalled_seconds=300.0,
    ):
        self.job = job
        self.status_callback = status_callback
        self.warning_callback = warning_callback
        self.clock = clock
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.stalled_seconds = float(stalled_seconds)
        self._last_log_at = {}
        self._last_event_at = None
        self._last_event = None

    def __call__(self, event):
        self._last_event = event
        now = self.clock()
        if (
            self._last_event_at is not None
            and now - self._last_event_at >= self.stalled_seconds
        ):
            _safe_warning(
                self.job,
                "Axis Search warning: progress resumed after "
                f"{now - self._last_event_at:.1f}s without a progress event; "
                f"stage={event.stage} family={event.family_name} "
                f"class={event.class_number} pass={event.pass_name}",
                self.warning_callback,
            )
        self._last_event_at = now
        key = (event.stage, event.family_name, event.class_number, event.pass_name)
        last_log_at = self._last_log_at.get(key)
        if (
            last_log_at is not None
            and event.completed != event.total
            and now - last_log_at < self.heartbeat_seconds
        ):
            return
        eta = "unknown" if event.eta_seconds is None else f"{event.eta_seconds:.1f}s"
        _safe_status(
            self.job,
            "Axis Search progress: "
            f"stage={event.stage} family={event.family_name} "
            f"class={event.class_number} pass={event.pass_name} "
            f"angles={event.completed}/{event.total} "
            f"evaluations={event.evaluation_count} "
            f"elapsed={event.elapsed_seconds:.1f}s eta={eta}",
            self.status_callback,
        )
        self._last_log_at[key] = now

    def context(self):
        if self._last_event is None:
            return "family=unknown class=unknown pass=unknown"
        return (
            f"family={self._last_event.family_name} "
            f"class={self._last_event.class_number} "
            f"pass={self._last_event.pass_name}"
        )


def _safe_status(job, message, callback):
    _safe_job_log(job, message)
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


def _safe_job_log(job, message):
    try:
        job.log(message)
    except Exception:
        pass


def _safe_warning(job, message, callback):
    _safe_status(job, message, callback)


def _attach_axis_dashboard_preview(job, preview_path, *, status_callback=None):
    for target, attach in (
        (
            "output card",
            lambda: job.set_output_image("axis_search_preview", preview_path),
        ),
        ("job tile", lambda: job.set_tile_image(preview_path)),
    ):
        try:
            attach()
        except Exception as error:
            _safe_warning(
                job,
                "WARNING: Could not attach Axis Search Dashboard Preview "
                f"to {target}; {type(error).__name__}: {error}",
                status_callback,
            )


def _native_axis_match(
    class_average,
    matching_map,
    rotation_matrix,
    *,
    reference_rotation_matrix=None,
    roll_degrees,
    class_pixel_size_A,
    map_pixel_size_A,
    score_config,
):
    grid = prepare_native_matching_grid(
        class_average,
        matching_map,
        class_pixel_size=class_pixel_size_A,
        volume_pixel_size=map_pixel_size_A,
    )
    projection = project_volume_at_rotation(grid.volume, rotation_matrix)
    reference_projection = (
        projection
        if reference_rotation_matrix is None
        else project_volume_at_rotation(grid.volume, reference_rotation_matrix)
    )
    rolled_projection = rotate_image(
        projection,
        float(roll_degrees),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shift_xy = find_projection_shift(grid.class_average, rolled_projection)
    matched_projection = shift_image(
        rolled_projection,
        shift=(shift_xy[1], shift_xy[0]),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    diagnostic = compute_diagnostic_band_limited_score(
        grid.class_average,
        matched_projection,
        pixel_size_A=grid.pixel_size,
        settings=score_config,
    )
    return {
        "projection": projection.astype(np.float32, copy=False),
        "reference_projection": reference_projection.astype(
            np.float32, copy=False
        ),
        "matched_projection": matched_projection,
        "shift_xy_pixels": [float(shift_xy[0]), float(shift_xy[1])],
        "pixel_size_A": float(grid.pixel_size),
        "diagnostic_score": {
            "score": diagnostic.score,
            "valid": diagnostic.valid,
            "invalid_reason": diagnostic.invalid_reason,
            **diagnostic.metadata,
        },
    }


def _load_templates(project, dataset):
    classes = {}
    pixel_sizes = set()
    stack_cache = {}
    for path_value, image_index, pixel_size in zip(
        dataset["blob/path"],
        dataset["blob/idx"],
        dataset["blob/psize_A"],
        strict=True,
    ):
        path = _resolve_project_path(project, path_value)
        if path not in stack_cache:
            _, stack_cache[path] = mrc.read(path)
        class_number = int(image_index) + 1
        if class_number in classes:
            raise ValueError(
                f"source Class Number {class_number} occurs more than once"
            )
        classes[class_number] = np.asarray(
            stack_cache[path][int(image_index)], dtype=np.float32
        )
        pixel_sizes.add(float(pixel_size))
    if not classes:
        raise ValueError("Select 2D templates input is empty")
    if len(pixel_sizes) != 1:
        raise ValueError("all class averages must share one native pixel size")
    return classes, pixel_sizes.pop()


def _add_input(job, *, name, type, slots, source, title):
    job.add_input(type=type, name=name, min=1, max=1, slots=slots, title=title)
    job.connect(name, source.job_uid, source.output_name)


def _save_template_stack(job, name, filename, stack, *, pixel_size_A):
    output = job.alloc_output(name, len(stack))
    output["blob/path"][:] = f">{job.uid}/{filename}"
    output["blob/idx"][:] = np.arange(len(stack))
    output["blob/shape"][:] = stack.shape[1:]
    output["blob/psize_A"][:] = pixel_size_A
    job.save_output(name, output)


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


def _resource_directory(resource):
    directory = resource.dir
    directory = directory() if callable(directory) else directory
    return Path(directory)


def _resolve_project_path(project, path):
    if isinstance(path, bytes):
        path = path.decode()
    path = Path(str(path).removeprefix(">"))
    if path.is_absolute():
        return path
    return _resource_directory(project) / path
