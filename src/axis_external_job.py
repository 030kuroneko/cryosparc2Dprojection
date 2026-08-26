"""Supported CryoSPARC External Job boundary for Symmetry-Axis Search."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.axis_search import (
    AxisProximityConfig,
    AxisSearchConfig,
    rank_axis_families,
    refine_axis_candidates,
)
from cryosparc_2d_projection.axis_presentation import (
    AXIS_RESULT_COLUMNS,
    AxisResultPanelRow,
    apply_axis_display_roll,
    create_axis_result_figure,
)
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    build_surface_model,
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
    output_names = (
        "axis_candidates_raw",
        "axis_candidates_aligned",
        "axis_near_projections",
        "axis_exact_references",
    )
    for name in output_names:
        job.add_output(
            type="template",
            name=name,
            slots=["blob"],
            title=name.replace("_", " ").title(),
        )

    with job.run():
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
        search_result = rank_axis_families(
            classes,
            matching_map,
            families=families,
            class_pixel_size_A=class_pixel_size_A,
            map_pixel_size_A=map_pixel_size_A,
            config=config,
        )
        refinement = refine_axis_candidates(
            search_result,
            matching_map,
            class_pixel_size_A=class_pixel_size_A,
            map_pixel_size_A=map_pixel_size_A,
            config=proximity_config,
        )
        rows = []
        raw_stack = []
        aligned_stack = []
        exact_stack = []
        near_stack = []
        panel_rows = []
        surface = build_surface_model(
            rendering_map,
            surface_level=render_options.surface_level,
            max_size=render_options.grid_size,
        )
        render_size = comparison_options.resolve_render_size(
            render_options.image_size
        )
        family_ranks = {name: 0 for name in search_result.families}
        for row_number, (candidate, refined) in enumerate(zip(
            search_result.rows, refinement.rows, strict=True
        ), start=1):
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
            exact_stack.append(
                apply_axis_display_roll(
                    candidate.exact_reference_display,
                    display_roll,
                    background=render_options.background,
                )
            )
            near_stack.append(
                apply_axis_display_roll(
                    refined.near_axis_projection_display,
                    display_roll,
                    background=render_options.background,
                )
            )
            exact_render_path = write_camera_view_render(
                _resource_directory(job) / "renders" / "exact",
                surface=surface,
                rotation_matrix=search_result.families[
                    candidate.family_name
                ].family.canonical_camera_matrix,
                class_number=row_number,
                image_size=render_size,
                background=render_options.background,
            )
            near_render_path = write_camera_view_render(
                _resource_directory(job) / "renders" / "near",
                surface=surface,
                rotation_matrix=refined.near_axis_rotation_matrix,
                class_number=row_number,
                image_size=render_size,
                background=render_options.background,
            )
            from PIL import Image

            with Image.open(near_render_path) as image:
                near_view = np.asarray(image.convert("RGB")).copy()
            with Image.open(exact_render_path) as image:
                exact_view = np.asarray(image.convert("RGB")).copy()
            panel_rows.append(
                AxisResultPanelRow(
                    family_name=candidate.family_name,
                    rank=rank,
                    class_number=candidate.class_number,
                    axis_aligned_class=candidate.aligned_class,
                    near_axis_projection=refined.near_axis_projection_display,
                    exact_axis_projection=candidate.exact_reference_display,
                    near_axis_view=near_view,
                    exact_axis_view=exact_view,
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
                    "score_provenance": candidate.score_metadata,
                    "refined_score": refined.refined_score,
                    "angular_distance_degrees": refined.angular_distance_degrees,
                    "exact_axis_rotation_matrix": candidate.exact_rotation_matrix.tolist(),
                    "near_axis_rotation_matrix": refined.near_axis_rotation_matrix.tolist(),
                    "axis_roll_degrees": display_roll,
                    "cone_boundary": refined.cone_boundary,
                    "duplicate": candidate.duplicate,
                    "warnings": list(candidate.warnings + refined.warnings),
                }
            )
        stacks = {
            "axis_candidates_raw": np.asarray(raw_stack, dtype=np.float32),
            "axis_candidates_aligned": np.asarray(aligned_stack, dtype=np.float32),
            "axis_near_projections": np.asarray(near_stack, dtype=np.float32),
            "axis_exact_references": np.asarray(exact_stack, dtype=np.float32),
        }
        job_directory = _resource_directory(job)
        for name, stack in stacks.items():
            filename = f"{name}.mrcs"
            mrc.write(job_directory / filename, stack, class_pixel_size_A)
            _save_template_stack(
                job,
                name,
                filename,
                stack,
                pixel_size_A=class_pixel_size_A,
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
                "cone_degrees": proximity_config.cone_degrees,
                "coarse_step_degrees": proximity_config.coarse_step_degrees,
                "refine_step_degrees": proximity_config.refine_step_degrees,
            },
            "presentation": {
                "columns": list(AXIS_RESULT_COLUMNS),
                "axis_rolls": axis_rolls,
                "comparison_dpi": comparison_options.dpi,
                "preview_page_size": comparison_options.page_size,
                "rendering_map": render_options.map_name,
                "render_size": render_size,
                "render_grid_size": render_options.grid_size,
                "surface_level": surface.surface_level,
                "background": render_options.background,
                "static_only": True,
            },
            "outputs": list(output_names),
            "rows": rows,
        }
        (job_directory / "axis_search_results.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
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
        for row in rows:
            job.log(
                "Axis Search row: "
                f"family={row['family']} rank={row['rank']} "
                f"class={row['class_number']} "
                f"exact_score={row['axis_class_score']:.6f} "
                f"refined_score={row['refined_score']:.6f} "
                f"angular_distance={row['angular_distance_degrees']:.3f} "
                f"duplicate={row['duplicate']} warnings={row['warnings']}"
            )
            job.log(
                "Axis Search row JSON: "
                + json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        job.log(
            f"Ranked {len(classes)} classes across "
            f"{len(search_result.families)} Axis Families."
        )
    return artifact


def _load_templates(project, dataset):
    classes = {}
    pixel_sizes = set()
    stack_cache = {}
    for index, (path_value, image_index, pixel_size) in enumerate(zip(
        dataset["blob/path"],
        dataset["blob/idx"],
        dataset["blob/psize_A"],
        strict=True,
    )):
        path = _resolve_project_path(project, path_value)
        if path not in stack_cache:
            _, stack_cache[path] = mrc.read(path)
        classes[index + 1] = np.asarray(
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
