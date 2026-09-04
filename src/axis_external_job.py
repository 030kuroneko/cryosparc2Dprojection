"""Symmetry-Axis Search workflow using the supported External Job adapter."""

import json
from time import monotonic

import numpy as np

from cryosparc_2d_projection.axis_search import (
    AxisProximityConfig,
    AxisSearchConfig,
    rank_axis_families,
    refine_axis_candidates,
)
from cryosparc_2d_projection.axis_result_rendering import (
    AxisResultRenderingEventCode,
    AxisResultRenderingRequest,
    render_axis_search_results,
)
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
)
from cryosparc_2d_projection.external_job_adapter import (
    CryoSPARCExternalJobAdapter,
    ExternalJobSource,
)


AxisSourceOutput = ExternalJobSource


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
    adapter = CryoSPARCExternalJobAdapter(
        project,
        workspace_uid,
        title="Symmetry-Axis Class Search (CryoSPARC 5.0.6)",
    )
    adapter.add_template_input(
        "templates",
        templates_source,
        title="Selected 2D class averages",
    )
    adapter.add_volume_input(
        "volume",
        volume_source,
        rendering_map=render_options.map_name,
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
        adapter.add_template_output(name, title=name.replace("_", " ").title())

    with adapter.run():
        run_started_at = monotonic()
        timings = {}
        adapter.set_status(
            "Axis Search stage: stage=input-loading status=started",
            status_callback,
        )
        stage_started_at = monotonic()
        templates = adapter.read_template_stack(
            "templates", require_unique=True, require_nonempty=True
        )
        classes = {
            class_number: template.image
            for class_number, template in templates.class_averages.items()
        }
        # Axis Search reports one-based Class Numbers while Dataset blob indices
        # are zero-based source indices.
        classes = {source_index + 1: image for source_index, image in classes.items()}
        class_pixel_size_A = templates.pixel_size_A
        volume_input = adapter.read_volume(
            "volume", rendering_map=render_options.map_name
        )
        matching_map = volume_input.matching_map
        map_pixel_size_A = volume_input.matching_pixel_size_A
        rendering_map = volume_input.rendering_map
        rendering_pixel_size_A = volume_input.rendering_pixel_size_A
        timings["input-loading"] = {
            "elapsed_seconds": monotonic() - stage_started_at
        }
        adapter.set_status(
            "Axis Search stage: stage=input-loading status=completed "
            f"elapsed={timings['input-loading']['elapsed_seconds']:.3f}s",
            status_callback,
        )
        adapter.set_status(
            "Axis Search stage: stage=exact-ranking status=started "
            f"classes={len(classes)} search_max_size={config.search_max_size}",
            status_callback,
        )
        stage_started_at = monotonic()
        progress_reporter = _AxisProgressReporter(
            adapter,
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
            adapter.set_status(
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
        adapter.set_status(
            "Axis Search stage: stage=exact-ranking status=completed "
            f"elapsed={timings['exact-ranking']['elapsed_seconds']:.3f}s",
            status_callback,
        )
        refinement = None
        if refine_near_axis:
            adapter.set_status(
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
                adapter.set_status(
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
            adapter.set_status(
                "Axis Search stage: stage=near-axis-refinement status=completed "
                f"elapsed={timings['near-axis-refinement']['elapsed_seconds']:.3f}s",
                status_callback,
            )
        def report_rendering_event(event):
            if event.code is AxisResultRenderingEventCode.RESULT_RENDERING_STARTED:
                adapter.set_status(
                    "Axis Search stage: stage=result-rendering status=started",
                    status_callback,
                )
            elif event.code is AxisResultRenderingEventCode.SURFACE_SAMPLING:
                adapter.set_status(event.message, status_callback)
            elif event.code is AxisResultRenderingEventCode.CANDIDATE_COMPLETED:
                adapter.set_status(
                    "Result Rendering progress: "
                    f"family={event.family_name} class={event.class_number} "
                    "status=completed",
                    status_callback,
                )
            elif event.code is AxisResultRenderingEventCode.OUTPUT_WRITING_STARTED:
                adapter.set_status(
                    "Axis Search stage: stage=output-writing status=started",
                    status_callback,
                )
            elif event.code is AxisResultRenderingEventCode.OUTPUT_WRITING_COMPLETED:
                adapter.set_status(
                    "Axis Search stage: stage=output-writing status=completed",
                    status_callback,
                )

        def report_rendering_warning(event):
            adapter.set_warning(event.message, warning_callback)

        rendering_started_at = monotonic()
        try:
            result = render_axis_search_results(
                AxisResultRenderingRequest(
                    output_directory=adapter.resource_directory,
                    search_result=search_result,
                    refinement=refinement,
                    matching_map=matching_map,
                    rendering_map=rendering_map,
                    class_pixel_size_A=class_pixel_size_A,
                    map_pixel_size_A=map_pixel_size_A,
                    rendering_pixel_size_A=rendering_pixel_size_A,
                    config=config,
                    proximity_config=proximity_config,
                    axis_rolls=axis_rolls,
                    comparison_options=comparison_options,
                    render_options=render_options,
                    refine_near_axis=refine_near_axis,
                    timings=timings,
                    progress_callback=report_rendering_event,
                    warning_callback=report_rendering_warning,
                    run_started_at=run_started_at,
                )
            )
        except Exception as error:
            adapter.set_status(
                "Axis Search stage: stage=result-rendering status=failed "
                f"elapsed={monotonic() - rendering_started_at:.3f}s "
                f"error={type(error).__name__}: {error}",
                status_callback,
            )
            raise
        artifact = result.artifact
        adapter.set_status(
            "Axis Search stage: stage=result-rendering status=completed "
            f"elapsed={artifact['timings']['result-rendering']['elapsed_seconds']:.3f}s",
            status_callback,
        )
        for name, stack in result.stacks.items():
            adapter.stage_template_stack(
                name,
                stack.filename,
                stack.data,
                pixel_size_A=stack.pixel_size_A,
            )
        adapter.publish()
        for page_number, page in enumerate(result.preview_pages, start=1):
            adapter.log_plot(
                page,
                f"Symmetry-Axis Search preview {page_number}/{len(result.preview_pages)}",
                formats=["png"],
                savefig_kw={"dpi": comparison_options.dpi, "bbox_inches": "tight"},
            )
        _attach_axis_dashboard_preview(
            adapter,
            result.preview_path,
            status_callback=status_callback,
        )
        for row in artifact["rows"]:
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
            adapter.set_status(
                "Axis Search row: "
                f"family={row['family']} rank={row['rank']} "
                f"class={row['class_number']} "
                f"exact_score={row['axis_class_score']:.6f} "
                f"refined_score={refined_text} "
                f"angular_distance={angular_text} "
                f"duplicate={row['duplicate']} warnings={row['warnings']}",
                status_callback,
            )
            adapter.safe_log(
                "Axis Search row JSON: "
                + json.dumps(row, sort_keys=True, separators=(",", ":")),
            )
        adapter.set_status(
            f"Ranked {len(classes)} classes across "
            f"{len(search_result.families)} Axis Families.",
            status_callback,
        )
    return artifact


class _AxisProgressReporter:
    def __init__(
        self,
        adapter,
        *,
        status_callback=None,
        warning_callback=None,
        clock=monotonic,
        heartbeat_seconds=30.0,
        stalled_seconds=300.0,
    ):
        self.adapter = adapter
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
            self.adapter.set_warning(
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
        self.adapter.set_status(
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

def _attach_axis_dashboard_preview(adapter, preview_path, *, status_callback=None):
    adapter.attach_output_preview(
        "axis_search_preview",
        preview_path,
        warning_callback=status_callback,
        warning_formatter=lambda error: (
            "WARNING: Could not attach Axis Search Dashboard Preview "
            f"to output card; {type(error).__name__}: {error}"
        ),
    )
    adapter.attach_tile_preview(
        preview_path,
        warning_callback=status_callback,
        warning_formatter=lambda error: (
            "WARNING: Could not attach Axis Search Dashboard Preview "
            f"to job tile; {type(error).__name__}: {error}"
        ),
    )
