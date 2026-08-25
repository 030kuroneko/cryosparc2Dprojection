from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.class_poses import analyze_class_orientations
from cryosparc_2d_projection.camera import solve_class_camera_from_particle_poses
from cryosparc_2d_projection.matching_grid import prepare_matching_grid
from cryosparc_2d_projection.projection import (
    project_native_matched_projection,
    rotate_volume_at_rotation,
)
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.scoring import (
    BandLimitedScoreConfig,
    compute_diagnostic_band_limited_score,
)
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    SurfaceRenderMemoryError,
    build_surface_model,
    resolve_surface_sampling_grid,
    write_camera_view_render,
)
from cryosparc_2d_projection.symmetry import SupportedSymmetry
from cryosparc_2d_projection.viewer import (
    create_class_preview_figure,
    create_class_preview_pages,
    write_chimerax_bundle,
)


TARGET_CRYOSPARC_VERSION = "5.0.6"


class NativeReprojectionError(RuntimeError):
    """A native Matched Projection could not be generated for a class."""


@dataclass(frozen=True)
class SourceOutput:
    job_uid: str
    output_name: str


@dataclass(frozen=True)
class LoadedClassAverage:
    image: np.ndarray
    pixel_size: float


def run_external_orientation_job(
    project,
    workspace_uid,
    select_2d_source,
    select_templates_source,
    refinement_source,
    volume_source,
    symmetry="C1",
    interactive_class_numbers=(),
    render_options=None,
    diagnostic_score_config=None,
    comparison_options=None,
    warning_callback=None,
    status_callback=None,
):
    """Create and run the CryoSPARC External Job for class orientation analysis."""
    symmetry = SupportedSymmetry.parse(symmetry).value
    render_options = render_options or ClassRenderOptions()
    diagnostic_score_config = (
        diagnostic_score_config or BandLimitedScoreConfig()
    )
    comparison_options = comparison_options or ComparisonRenderOptions()
    rendering_slot = (
        "map_sharp" if render_options.map_name == "sharpened" else "map"
    )
    job = project.create_external_job(
        workspace_uid,
        title=f"2D Class Orientation (CryoSPARC {TARGET_CRYOSPARC_VERSION})",
    )
    _add_and_connect_input(
        job,
        name="select_2d_templates",
        type="template",
        slots=["blob"],
        source=select_templates_source,
        title="Selected 2D class averages",
    )
    _add_and_connect_input(
        job,
        name="select_2d_particles",
        type="particle",
        slots=["alignments2D"],
        source=select_2d_source,
        title="Select 2D particles",
    )
    _add_and_connect_input(
        job,
        name="refinement_particles",
        type="particle",
        slots=["alignments3D"],
        source=refinement_source,
        title="NU or Local Refinement particles",
    )
    _add_and_connect_input(
        job,
        name="refinement_volume",
        type="volume",
        slots=(
            ["map", "map_sharp"]
            if render_options.map_name == "sharpened"
            else ["map"]
        ),
        source=volume_source,
        title="NU or Local Refinement volume",
    )
    job.add_output(
        type="template",
        name="matched_projections",
        slots=["blob"],
        title="Matched class projections",
    )
    job.add_output(
        type="template",
        name="search_projections",
        slots=["blob"],
        title="Bounded camera-search projections",
    )
    job.add_output(
        type="volume",
        name="rendering_map",
        slots=["map"],
        title="Rendering map",
    )
    for class_number in interactive_class_numbers or ():
        job.add_output(
            type="volume",
            name=f"class_{class_number:03d}_volume",
            slots=["map"],
            title=f"Class {class_number} interactive volume",
        )

    with job.run():
        orientations = analyze_class_orientations(
            job.load_input("select_2d_particles"),
            job.load_input("refinement_particles"),
            symmetry=symmetry,
        )
        resolved_presentation = comparison_options.resolve(
            class_count=len(orientations),
            requested_render_size=render_options.image_size,
        )
        for warning in resolved_presentation.warnings:
            job.log(warning)
            if warning_callback is not None:
                warning_callback(warning)
        class_averages = _load_class_averages(
            project, job.load_input("select_2d_templates")
        )
        _validate_native_class_grids(class_averages, orientations)
        artifact = {
            "cryosparc_version": TARGET_CRYOSPARC_VERSION,
            "symmetry": symmetry,
            "classes": [
                {
                    "class_id": class_id,
                    "class_number": class_id + 1,
                    "particle_count": orientation.particle_count,
                    "view_direction": orientation.view_direction.tolist(),
                    "angular_spread_degrees": orientation.angular_spread_degrees,
                }
                for class_id, orientation in sorted(orientations.items())
            ],
            "presentation": {
                "comparison_dpi": resolved_presentation.comparison_dpi,
                "preview_page_size": resolved_presentation.preview_page_size,
                "requested_render_size": resolved_presentation.requested_render_size,
                "effective_render_size": resolved_presentation.effective_render_size,
                "render_size_was_automatic": (
                    resolved_presentation.render_size_was_automatic
                ),
                "estimated_page_width_px": (
                    resolved_presentation.estimated_page_width_px
                ),
                "estimated_page_height_px": (
                    resolved_presentation.estimated_page_height_px
                ),
                "estimated_page_rgba_memory_bytes": (
                    resolved_presentation.estimated_page_rgba_memory_bytes
                ),
                "warnings": list(resolved_presentation.warnings),
            },
        }
        job_directory = Path(_directory_of(job))
        output_path = job_directory / "class_orientations.json"
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")

        volume_input = job.load_input("refinement_volume")
        volume_path = _resolve_project_path(project, volume_input["map/path"][0])
        _, volume_data = mrc.read(volume_path)
        pixel_size = float(volume_input["map/psize_A"][0])
        rendering_volume_path = _resolve_project_path(
            project, volume_input[f"{rendering_slot}/path"][0]
        )
        _, rendering_volume_data = mrc.read(rendering_volume_path)
        rendering_pixel_size = float(
            volume_input[f"{rendering_slot}/psize_A"][0]
        )
        rendering_output = job.alloc_output("rendering_map", 1)
        rendering_output["map/path"][:] = volume_input[f"{rendering_slot}/path"][0]
        rendering_output["map/shape"][:] = rendering_volume_data.shape
        rendering_output["map/psize_A"][:] = rendering_pixel_size
        job.save_output("rendering_map", rendering_output)
        select_particles = job.load_input("select_2d_particles")
        refinement_particles = job.load_input("refinement_particles")
        camera_results = {}
        native_projection_results = {}
        diagnostic_scores = {}
        matching_grids = {}
        for class_id in sorted(orientations):
            refinement_poses, alignment_2d_poses = _matched_particle_poses(
                select_particles, refinement_particles, class_id
            )
            template = class_averages[class_id]
            matching_grid = prepare_matching_grid(
                template.image,
                volume_data,
                class_pixel_size=template.pixel_size,
                volume_pixel_size=pixel_size,
                max_size=128,
            )
            matching_grids[class_id] = matching_grid
            camera_results[class_id] = solve_class_camera_from_particle_poses(
                matching_grid.class_average,
                matching_grid.volume,
                refinement_poses=refinement_poses,
                alignment_2d_poses=alignment_2d_poses,
                symmetry=symmetry,
            )
            try:
                native_projection_results[class_id] = (
                    project_native_matched_projection(
                        template.image,
                        volume_data,
                        camera_results[class_id].rotation_matrix,
                        class_pixel_size=template.pixel_size,
                        volume_pixel_size=pixel_size,
                    )
                )
            except (MemoryError, RuntimeError, ValueError) as error:
                failure = NativeReprojectionError(
                    f"Native Matched Projection failed for Class {class_id + 1}; "
                    "the bounded Search Projection was not substituted. "
                    f"Cause: {error}"
                )
                job.log(str(failure))
                raise failure from error
            diagnostic_scores[class_id] = (
                compute_diagnostic_band_limited_score(
                    template.image,
                    native_projection_results[class_id].matched_projection,
                    pixel_size_A=template.pixel_size,
                    settings=diagnostic_score_config,
                )
            )
        sampling_grid = resolve_surface_sampling_grid(
            rendering_volume_data.shape,
            render_options.grid_size,
        )
        original_shape = " x ".join(
            str(size) for size in sampling_grid.original_shape
        )
        sampling_shape = " x ".join(
            str(size) for size in sampling_grid.sampled_shape
        )
        requested_grid = (
            "native"
            if sampling_grid.requested_grid_size is None
            else str(sampling_grid.requested_grid_size)
        )
        sampling_message = (
            f"Surface Sampling Grid: original={original_shape}; "
            f"requested={requested_grid}; effective={sampling_shape}; "
            f"mode={sampling_grid.mode}; downsampled="
            f"{'yes' if sampling_grid.was_downsampled else 'no'}; "
            "estimated minimum working memory="
            f"{sampling_grid.estimated_memory_gib:.3f} GiB "
            "(mesh and plotting allocations excluded); "
            f"Camera View Render={resolved_presentation.effective_render_size} px."
        )
        job.log(sampling_message)
        if status_callback is not None:
            status_callback(sampling_message)
        for warning in sampling_grid.warnings:
            job.log(warning)
            if warning_callback is not None:
                warning_callback(warning)
        try:
            surface = build_surface_model(
                rendering_volume_data,
                surface_level=render_options.surface_level,
                sampling_grid=sampling_grid,
            )
        except SurfaceRenderMemoryError as error:
            job.log(str(error))
            raise
        job.log(f"Surface Level: {surface.surface_level:.6g}")
        if surface.warning:
            job.log(surface.warning)
        artifact["rendering"] = {
            "map": render_options.map_name,
            "surface_level": surface.surface_level,
            "surface_level_was_automatic": surface.surface_level_was_automatic,
            "warning": surface.warning,
            "background": render_options.background,
            "image_size": resolved_presentation.effective_render_size,
            "grid_size": sampling_grid.effective_grid_size,
            **sampling_grid.as_dict(),
        }
        render_paths = {}
        for class_entry in artifact["classes"]:
            camera = camera_results[class_entry["class_id"]]
            native_projection = native_projection_results[class_entry["class_id"]]
            template = class_averages[class_entry["class_id"]]
            diagnostic = diagnostic_scores[class_entry["class_id"]]
            diagnostic_metadata = {
                key: value
                for key, value in diagnostic.metadata.items()
                if key
                not in {
                    "band_limited_score_valid",
                    "band_limited_invalid_reason",
                }
            }
            class_entry["camera"] = {
                "rotation_matrix": camera.rotation_matrix.tolist(),
                "quaternion_xyzw": camera.quaternion_xyzw.tolist(),
                "view_direction": camera.view_direction.tolist(),
                "in_plane_rotation_degrees": camera.in_plane_rotation_degrees,
                "projection_shift_pixels": (
                    native_projection.projection_shift_pixels.tolist()
                ),
                "search_projection_shift_pixels": (
                    camera.projection_shift_pixels.tolist()
                ),
                "match_score": camera.match_score,
                "second_best_score": camera.second_best_score,
                "score_margin": camera.score_margin,
                "match_confidence": camera.match_confidence,
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
                "search_box_size": int(
                    matching_grids[class_entry["class_id"]].class_average.shape[0]
                ),
                "search_pixel_size_A": matching_grids[
                    class_entry["class_id"]
                ].pixel_size,
                "matching_box_size": int(template.image.shape[0]),
                "matching_pixel_size_A": template.pixel_size,
                "search_evaluation_count": camera.search_evaluation_count,
                "coordinate_convention": (
                    "right-handed Cartesian active rotation; "
                    "image rows increase downward"
                ),
            }
            try:
                render_paths[class_entry["class_id"]] = write_camera_view_render(
                    job_directory / "renders",
                    surface=surface,
                    rotation_matrix=camera.rotation_matrix,
                    class_number=class_entry["class_number"],
                    image_size=resolved_presentation.effective_render_size,
                    background=render_options.background,
                )
            except SurfaceRenderMemoryError as error:
                job.log(str(error))
                raise
        write_chimerax_bundle(
            job_directory / "chimerax",
            map_path=str(rendering_volume_path),
            cameras=camera_results,
        )
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")
        projections = np.asarray(
            [
                native_projection_results[class_id].matched_projection
                for class_id in sorted(orientations)
            ],
            dtype=np.float32,
        )
        search_projections = np.asarray(
            [
                camera_results[class_id].matched_projection
                for class_id in sorted(orientations)
            ],
            dtype=np.float32,
        )
        first_class_id = sorted(orientations)[0]
        projection_pixel_size = class_averages[first_class_id].pixel_size
        search_projection_pixel_size = matching_grids[first_class_id].pixel_size
        mrc.write(
            job_directory / "class_projections.mrcs",
            projections,
            projection_pixel_size,
        )
        mrc.write(
            job_directory / "search_projections.mrcs",
            search_projections,
            search_projection_pixel_size,
        )
        for class_number in interactive_class_numbers or ():
            class_id = class_number - 1
            if class_id not in camera_results:
                raise ValueError(f"Class {class_number} is not present in selected classes")
            name = f"class_{class_number:03d}_volume"
            rotated_volume = rotate_volume_at_rotation(
                rendering_volume_data, camera_results[class_id].rotation_matrix
            ).astype(np.float32, copy=False)
            filename = f"{name}.mrc"
            mrc.write(
                job_directory / filename, rotated_volume, rendering_pixel_size
            )
            volume_output = job.alloc_output(name, 1)
            volume_output["map/path"][:] = f">{job.uid}/{filename}"
            volume_output["map/shape"][:] = rotated_volume.shape
            volume_output["map/psize_A"][:] = rendering_pixel_size
            job.save_output(name, volume_output)
        preview_pages = create_class_preview_pages(
            class_averages,
            projections,
            camera_results,
            orientations,
            render_paths,
            diagnostic_scores=diagnostic_scores,
            comparison_options=comparison_options,
        )
        for class_id in sorted(orientations):
            comparison = create_class_preview_figure(
                class_averages,
                projections,
                camera_results,
                orientations,
                render_paths,
                diagnostic_scores=diagnostic_scores,
                comparison_options=comparison_options,
                class_ids=[class_id],
            )
            comparison.savefig(
                job_directory
                / "renders"
                / f"class_{class_id + 1:03d}_comparison.png",
                dpi=comparison_options.dpi,
            )
        preview = preview_pages[0]
        projection_output = job.alloc_output("matched_projections", len(projections))
        projection_output["blob/path"][:] = f">{job.uid}/class_projections.mrcs"
        projection_output["blob/idx"][:] = np.arange(len(projections))
        projection_output["blob/shape"][:] = projections.shape[1:]
        projection_output["blob/psize_A"][:] = projection_pixel_size
        job.save_output("matched_projections", projection_output, image=preview)
        search_projection_output = job.alloc_output(
            "search_projections", len(search_projections)
        )
        search_projection_output["blob/path"][:] = (
            f">{job.uid}/search_projections.mrcs"
        )
        search_projection_output["blob/idx"][:] = np.arange(
            len(search_projections)
        )
        search_projection_output["blob/shape"][:] = search_projections.shape[1:]
        search_projection_output["blob/psize_A"][:] = search_projection_pixel_size
        job.save_output("search_projections", search_projection_output)
        for page_number, page in enumerate(preview_pages, start=1):
            job.log_plot(
                page,
                f"Class camera preview {page_number}/{len(preview_pages)}",
                formats=["png"],
                savefig_kw={
                    "dpi": comparison_options.dpi,
                    "bbox_inches": "tight",
                    "pad_inches": 0,
                },
            )
        job.log(
            f"Analyzed {len(orientations)} 2D classes using overlapping particle UIDs."
        )

    return job


def _load_class_averages(project, templates):
    stacks = {}
    class_averages = {}
    for path, index, pixel_size in zip(
        templates["blob/path"],
        templates["blob/idx"],
        templates["blob/psize_A"],
        strict=True,
    ):
        resolved = _resolve_project_path(project, path)
        if resolved not in stacks:
            _, stacks[resolved] = mrc.read(resolved)
        stack = stacks[resolved]
        image = stack[int(index)] if stack.ndim == 3 else stack
        class_averages[int(index)] = LoadedClassAverage(
            image=np.asarray(image), pixel_size=float(pixel_size)
        )
    return class_averages


def _validate_native_class_grids(class_averages, orientations):
    shapes = {
        class_id: tuple(class_averages[class_id].image.shape)
        for class_id in sorted(orientations)
    }
    if len(set(shapes.values())) > 1:
        details = "; ".join(
            f"Class {class_id + 1} {shape}" for class_id, shape in shapes.items()
        )
        raise ValueError(
            "Native Class Average boxes must have one common shape: " + details
        )

    pixel_sizes = {
        class_id: class_averages[class_id].pixel_size
        for class_id in sorted(orientations)
    }
    reference_pixel_size = next(iter(pixel_sizes.values()))
    if all(
        np.isclose(pixel_size, reference_pixel_size)
        for pixel_size in pixel_sizes.values()
    ):
        return
    details = "; ".join(
        f"Class {class_id + 1} {pixel_size:g} A/pixel"
        for class_id, pixel_size in pixel_sizes.items()
    )
    raise ValueError(
        "Native Class Averages must have one common pixel size for MRCS output: "
        + details
    )


def _matched_particle_poses(select_particles, refinement_particles, class_id):
    refinement_rows = {
        int(uid): row for row, uid in enumerate(refinement_particles["uid"])
    }
    poses_3d = []
    poses_2d = []
    for row, (uid, particle_class) in enumerate(
        zip(
            select_particles["uid"],
            select_particles["alignments2D/class"],
            strict=True,
        )
    ):
        refinement_row = refinement_rows.get(int(uid))
        if int(particle_class) != class_id or refinement_row is None:
            continue
        poses_3d.append(refinement_particles["alignments3D/pose"][refinement_row])
        poses_2d.append(select_particles["alignments2D/pose"][row])
    return np.asarray(poses_3d), np.asarray(poses_2d)


def _add_and_connect_input(job, *, name, type, slots, source, title):
    job.add_input(
        type=type,
        name=name,
        min=1,
        max=1,
        slots=slots,
        title=title,
    )
    job.connect(name, source.job_uid, source.output_name)


def _directory_of(resource):
    directory = resource.dir
    return directory() if callable(directory) else directory


def _resolve_project_path(project, path):
    if isinstance(path, bytes):
        path = path.decode()
    path = Path(str(path).removeprefix(">"))
    if path.is_absolute():
        return path
    return Path(_directory_of(project)) / path
