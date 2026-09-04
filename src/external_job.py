import json

import numpy as np

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
    get_surface_camera_viewport_A,
    resolve_surface_sampling_grid,
    write_camera_view_render,
)
from cryosparc_2d_projection.auto_crop import (
    PhysicalCameraView,
    compute_auto_crop_2d_framing,
)
from cryosparc_2d_projection.symmetry import SupportedSymmetry
from cryosparc_2d_projection.viewer import (
    create_class_preview_figure,
    create_class_preview_pages,
    write_matched_projection_thumbnail,
    write_chimerax_bundle,
)
from cryosparc_2d_projection.external_job_adapter import (
    CryoSPARCExternalJobAdapter,
    ExternalJobSource,
    TARGET_CRYOSPARC_VERSION,
)


class NativeReprojectionError(RuntimeError):
    """A native Matched Projection could not be generated for a class."""


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
    diagnostic_score_config = (
        diagnostic_score_config or BandLimitedScoreConfig()
    )
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
        select_particles = adapter.read_2d_particle_alignments(
            "select_2d_particles"
        )
        refinement_particles = adapter.read_3d_particle_alignments(
            "refinement_particles"
        )
        orientations = analyze_class_orientations(
            select_particles, refinement_particles, symmetry=symmetry
        )
        resolved_presentation = comparison_options.resolve(
            class_count=len(orientations),
            requested_render_size=render_options.image_size,
        )
        for warning in resolved_presentation.warnings:
            adapter.log(warning)
            if warning_callback is not None:
                warning_callback(warning)
        class_averages = adapter.read_template_stack(
            "select_2d_templates"
        ).class_averages
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
        if comparison_options.auto_crop_2d:
            artifact["presentation"]["auto_crop_2d"] = {
                "enabled": True,
                "mode": "physical_camera_fov",
            }
        job_directory = adapter.resource_directory
        output_path = job_directory / "class_orientations.json"
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")

        volume_input = adapter.read_volume(
            "refinement_volume", rendering_map=render_options.map_name
        )
        volume_data = volume_input.matching_map
        pixel_size = volume_input.matching_pixel_size_A
        rendering_volume_path = volume_input.rendering_path
        rendering_volume_data = volume_input.rendering_map
        rendering_pixel_size = volume_input.rendering_pixel_size_A
        adapter.stage_volume_source(
            "rendering_map",
            rendering_volume_path,
            shape=rendering_volume_data.shape,
            pixel_size_A=rendering_pixel_size,
            dataset_path=volume_input.rendering_dataset_path,
        )
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
                class_pixel_size=template.pixel_size_A,
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
                        class_pixel_size=template.pixel_size_A,
                        volume_pixel_size=pixel_size,
                    )
                )
            except (MemoryError, RuntimeError, ValueError) as error:
                failure = NativeReprojectionError(
                    f"Native Matched Projection failed for Class {class_id + 1}; "
                    "the bounded Search Projection was not substituted. "
                    f"Cause: {error}"
                )
                adapter.log(str(failure))
                raise failure from error
            diagnostic_scores[class_id] = (
                compute_diagnostic_band_limited_score(
                    template.image,
                    native_projection_results[class_id].matched_projection,
                    pixel_size_A=template.pixel_size_A,
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
        adapter.log(sampling_message)
        if status_callback is not None:
            status_callback(sampling_message)
        for warning in sampling_grid.warnings:
            adapter.log(warning)
            if warning_callback is not None:
                warning_callback(warning)
        try:
            surface = build_surface_model(
                rendering_volume_data,
                surface_level=render_options.surface_level,
                sampling_grid=sampling_grid,
            )
        except SurfaceRenderMemoryError as error:
            adapter.log(str(error))
            raise
        adapter.log(f"Surface Level: {surface.surface_level:.6g}")
        if surface.warning:
            adapter.log(surface.warning)
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
        camera_viewport_A = None
        camera_viewport_error = None
        if comparison_options.auto_crop_2d:
            try:
                camera_viewport_A = get_surface_camera_viewport_A(
                    surface,
                    rendering_pixel_size_A=rendering_pixel_size,
                )
                camera_viewport_A = PhysicalCameraView(
                    camera_viewport_A=camera_viewport_A
                ).camera_viewport_A
            except (TypeError, ValueError, OverflowError) as error:
                camera_viewport_error = error
                warning_message = (
                    "WARNING: Auto-Cropped 2D Framing fell back for all classes: "
                    f"invalid physical camera viewport ({error})"
                )
                adapter.log(warning_message)
                if warning_callback is not None:
                    try:
                        warning_callback(warning_message)
                    except Exception:
                        pass
            else:
                artifact["presentation"]["auto_crop_2d"][
                    "camera_viewport_A"
                ] = float(camera_viewport_A)
        render_paths = {}
        framing_decisions = {}
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
                "matching_pixel_size_A": template.pixel_size_A,
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
                adapter.log(str(error))
                raise
            if comparison_options.auto_crop_2d:
                class_id = class_entry["class_id"]
                template = class_averages[class_id]
                native_projection = native_projection_results[class_id]
                camera_view = None
                if camera_viewport_error is None:
                    camera_view = PhysicalCameraView(
                        camera_viewport_A=camera_viewport_A,
                        projection_shift_pixels=tuple(
                            native_projection.projection_shift_pixels
                        ),
                    )
                decision = compute_auto_crop_2d_framing(
                    native_projection.matched_projection.shape,
                    template.pixel_size_A,
                    [] if camera_view is None else [camera_view],
                    enabled=True,
                )
                if decision.fallback:
                    reason = decision.fallback_reason
                    if camera_viewport_error is None:
                        warning_message = (
                            "WARNING: Auto-Cropped 2D Framing fell back for "
                            f"Class {class_entry['class_number']}: {reason}"
                        )
                        adapter.log(warning_message)
                        if warning_callback is not None:
                            try:
                                warning_callback(warning_message)
                            except Exception:
                                pass
                framing_decisions[class_id] = decision
                framing_metadata = decision.as_dict()
                framing_metadata["camera_view"] = (
                    None if camera_view is None else camera_view.as_dict()
                )
                class_entry["presentation"] = {
                    "auto_crop_2d": framing_metadata
                }
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
        projection_pixel_size = class_averages[first_class_id].pixel_size_A
        search_projection_pixel_size = matching_grids[first_class_id].pixel_size
        adapter.stage_template_stack(
            "matched_projections",
            "class_projections.mrcs",
            projections,
            pixel_size_A=projection_pixel_size,
        )
        adapter.stage_template_stack(
            "search_projections",
            "search_projections.mrcs",
            search_projections,
            pixel_size_A=search_projection_pixel_size,
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
            adapter.stage_volume(
                name,
                filename,
                rotated_volume,
                pixel_size_A=rendering_pixel_size,
            )
        adapter.publish()
        preview_pages = create_class_preview_pages(
            class_averages,
            projections,
            camera_results,
            orientations,
            render_paths,
            diagnostic_scores=diagnostic_scores,
            comparison_options=comparison_options,
            auto_crop_decisions=framing_decisions,
        )
        comparison_paths = {
            class_id: job_directory
            / "renders"
            / f"class_{class_id + 1:03d}_comparison.png"
            for class_id in sorted(orientations)
        }
        for class_id, comparison_path in comparison_paths.items():
            comparison = create_class_preview_figure(
                class_averages,
                projections,
                camera_results,
                orientations,
                render_paths,
                diagnostic_scores=diagnostic_scores,
                comparison_options=comparison_options,
                class_ids=[class_id],
                auto_crop_decisions=framing_decisions,
            )
            comparison.savefig(comparison_path, dpi=comparison_options.dpi)
        thumbnail_path = write_matched_projection_thumbnail(
            job_directory / "renders" / "matched_projections_thumbnail.png",
            projections[0],
        )
        adapter.attach_output_preview(
            "matched_projections",
            thumbnail_path,
            warning_formatter=lambda error: (
                "WARNING: Could not attach matched_projections thumbnail; "
                f"scientific output remains available. {error}"
            ),
        )
        adapter.attach_tile_preview(
            comparison_paths[first_class_id],
            warning_formatter=lambda error: (
                "WARNING: Could not attach Class Orientation Dashboard Preview "
                "to job tile; scientific output remains available. "
                f"{type(error).__name__}: {error}"
            ),
        )
        for page_number, page in enumerate(preview_pages, start=1):
            adapter.log_plot(
                page,
                f"Class camera preview {page_number}/{len(preview_pages)}",
                formats=["png"],
                savefig_kw={
                    "dpi": comparison_options.dpi,
                    "bbox_inches": "tight",
                    "pad_inches": 0,
                },
            )
        adapter.log(
            f"Analyzed {len(orientations)} 2D classes using overlapping particle UIDs."
        )

    return adapter.job


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
        class_id: class_averages[class_id].pixel_size_A
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
        int(uid): row for row, uid in enumerate(refinement_particles.uids)
    }
    poses_3d = []
    poses_2d = []
    for row, (uid, particle_class) in enumerate(
        zip(
            select_particles.uids,
            select_particles.class_ids,
            strict=True,
        )
    ):
        refinement_row = refinement_rows.get(int(uid))
        if int(particle_class) != class_id or refinement_row is None:
            continue
        poses_3d.append(refinement_particles.poses[refinement_row])
        poses_2d.append(select_particles.poses[row])
    return np.asarray(poses_3d), np.asarray(poses_2d)
