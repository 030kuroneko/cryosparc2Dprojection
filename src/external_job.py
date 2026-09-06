import numpy as np

from cryosparc_2d_projection.camera import solve_class_camera_from_particle_poses
from cryosparc_2d_projection.class_poses import analyze_class_orientations
from cryosparc_2d_projection.class_result_rendering import (
    ClassResultInput,
    ClassResultRenderingRequest,
    NativeReprojectionError,
    render_class_results,
)
from cryosparc_2d_projection.external_job_adapter import (
    CryoSPARCExternalJobAdapter,
    ExternalJobSource,
    TARGET_CRYOSPARC_VERSION,
)
from cryosparc_2d_projection.matching_grid import (
    prepare_matching_grid,
    validate_native_class_grids,
)
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.projection import rotate_volume_at_rotation
from cryosparc_2d_projection.scoring import BandLimitedScoreConfig
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    SurfaceRenderMemoryError,
)
from cryosparc_2d_projection.symmetry import SupportedSymmetry
from cryosparc_2d_projection.viewer import write_chimerax_bundle


SourceOutput = ExternalJobSource


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
    diagnostic_score_config = diagnostic_score_config or BandLimitedScoreConfig()
    comparison_options = comparison_options or ComparisonRenderOptions()
    adapter = CryoSPARCExternalJobAdapter(
        project,
        workspace_uid,
        title=f"2D Class Orientation (CryoSPARC {TARGET_CRYOSPARC_VERSION})",
    )
    adapter.add_template_input(
        "select_2d_templates",
        select_templates_source,
        title="Selected 2D class averages",
    )
    adapter.add_2d_particle_input(
        "select_2d_particles",
        select_2d_source,
        title="Select 2D particles",
    )
    adapter.add_3d_particle_input(
        "refinement_particles",
        refinement_source,
        title="NU or Local Refinement particles",
    )
    adapter.add_volume_input(
        "refinement_volume",
        volume_source,
        rendering_map=render_options.map_name,
        title="NU or Local Refinement volume",
    )
    adapter.add_template_output(
        "matched_projections", title="Matched class projections"
    )
    adapter.add_template_output(
        "search_projections", title="Bounded camera-search projections"
    )
    adapter.add_volume_output("rendering_map", title="Rendering map")
    for class_number in interactive_class_numbers or ():
        adapter.add_volume_output(
            f"class_{class_number:03d}_volume",
            title=f"Class {class_number} interactive volume",
        )

    with adapter.run():
        select_particles = adapter.read_2d_particle_alignments("select_2d_particles")
        refinement_particles = adapter.read_3d_particle_alignments(
            "refinement_particles"
        )
        orientations = analyze_class_orientations(
            select_particles, refinement_particles, symmetry=symmetry
        )
        class_averages = adapter.read_template_stack(
            "select_2d_templates"
        ).class_averages
        validate_native_class_grids(class_averages, orientations)
        volume_input = adapter.read_volume(
            "refinement_volume", rendering_map=render_options.map_name
        )
        adapter.stage_volume_source(
            "rendering_map",
            volume_input.rendering_path,
            shape=volume_input.rendering_map.shape,
            pixel_size_A=volume_input.rendering_pixel_size_A,
            dataset_path=volume_input.rendering_dataset_path,
        )

        camera_results = {}
        result_inputs = []
        for class_id in sorted(orientations):
            refinement_poses, alignment_2d_poses = _matched_particle_poses(
                select_particles, refinement_particles, class_id
            )
            template = class_averages[class_id]
            matching_grid = prepare_matching_grid(
                template.image,
                volume_input.matching_map,
                class_pixel_size=template.pixel_size_A,
                volume_pixel_size=volume_input.matching_pixel_size_A,
                max_size=128,
            )
            camera = solve_class_camera_from_particle_poses(
                matching_grid.class_average,
                matching_grid.volume,
                refinement_poses=refinement_poses,
                alignment_2d_poses=alignment_2d_poses,
                symmetry=symmetry,
            )
            camera_results[class_id] = camera
            result_inputs.append(
                ClassResultInput(
                    class_id=class_id,
                    class_average=template,
                    orientation=orientations[class_id],
                    camera=camera,
                    search_projection=camera.matched_projection,
                    search_pixel_size_A=matching_grid.pixel_size,
                )
            )

        def report_progress(event):
            adapter.log(event.message)
            if event.stage == "surface-sampling" and status_callback is not None:
                status_callback(event.message)

        def report_warning(event):
            adapter.log(event.message)
            if warning_callback is not None:
                warning_callback(event.message)

        try:
            result_set = render_class_results(
                ClassResultRenderingRequest(
                    output_directory=adapter.resource_directory,
                    classes=tuple(result_inputs),
                    matching_map=volume_input.matching_map,
                    matching_pixel_size_A=volume_input.matching_pixel_size_A,
                    rendering_map=volume_input.rendering_map,
                    rendering_pixel_size_A=volume_input.rendering_pixel_size_A,
                    symmetry=symmetry,
                    render_options=render_options,
                    comparison_options=comparison_options,
                    diagnostic_score_config=diagnostic_score_config,
                    progress_callback=report_progress,
                    warning_callback=report_warning,
                    cryosparc_version=TARGET_CRYOSPARC_VERSION,
                )
            )
        except (NativeReprojectionError, SurfaceRenderMemoryError) as error:
            adapter.log(str(error))
            raise

        for name, stack in result_set.stacks.items():
            adapter.stage_template_source(
                name,
                stack.path.name,
                count=stack.count,
                shape=stack.shape,
                pixel_size_A=stack.pixel_size_A,
            )

        write_chimerax_bundle(
            adapter.resource_directory / "chimerax",
            map_path=str(volume_input.rendering_path),
            cameras=camera_results,
        )
        for class_number in interactive_class_numbers or ():
            class_id = class_number - 1
            if class_id not in camera_results:
                raise ValueError(
                    f"Class {class_number} is not present in selected classes"
                )
            name = f"class_{class_number:03d}_volume"
            rotated_volume = rotate_volume_at_rotation(
                volume_input.rendering_map,
                camera_results[class_id].rotation_matrix,
            ).astype(np.float32, copy=False)
            adapter.stage_volume(
                name,
                f"{name}.mrc",
                rotated_volume,
                pixel_size_A=volume_input.rendering_pixel_size_A,
            )
        adapter.publish()

        first_class_id = min(camera_results)
        adapter.attach_output_preview(
            "matched_projections",
            result_set.thumbnail_path,
            warning_formatter=lambda error: (
                "WARNING: Could not attach matched_projections thumbnail; "
                f"scientific output remains available. {error}"
            ),
        )
        adapter.attach_tile_preview(
            result_set.comparison_paths[first_class_id],
            warning_formatter=lambda error: (
                "WARNING: Could not attach Class Orientation Dashboard Preview "
                "to job tile; scientific output remains available. "
                f"{type(error).__name__}: {error}"
            ),
        )
        for page_number, path in enumerate(result_set.preview_paths, start=1):
            adapter.log_plot_image(
                path,
                f"Class camera preview {page_number}/{len(result_set.preview_paths)}",
                formats=["png"],
                dpi=comparison_options.dpi,
            )
        adapter.log(
            f"Analyzed {len(orientations)} 2D classes using overlapping particle UIDs."
        )

    return adapter.job


def _matched_particle_poses(select_particles, refinement_particles, class_id):
    refinement_rows = {
        int(uid): row for row, uid in enumerate(refinement_particles.uids)
    }
    poses_3d = []
    poses_2d = []
    for row, (uid, particle_class) in enumerate(
        zip(select_particles.uids, select_particles.class_ids, strict=True)
    ):
        refinement_row = refinement_rows.get(int(uid))
        if int(particle_class) != class_id or refinement_row is None:
            continue
        poses_3d.append(refinement_particles.poses[refinement_row])
        poses_2d.append(select_particles.poses[row])
    return np.asarray(poses_3d), np.asarray(poses_2d)
