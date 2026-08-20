from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from cryosparc import mrc

from cryosparc_2d_projection.class_poses import analyze_class_orientations
from cryosparc_2d_projection.camera import solve_class_camera_from_particle_poses
from cryosparc_2d_projection.matching_grid import prepare_matching_grid
from cryosparc_2d_projection.projection import rotate_volume_at_rotation
from cryosparc_2d_projection.viewer import write_chimerax_bundle
from cryosparc_2d_projection.symmetry import assign_symmetry_axis


TARGET_CRYOSPARC_VERSION = "5.0.6"


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
):
    """Create and run the CryoSPARC External Job for class orientation analysis."""
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
        slots=["map"],
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
        type="volume",
        name="rendering_map",
        slots=["map"],
        passthrough="refinement_volume",
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
        }
        job_directory = Path(_directory_of(job))
        output_path = job_directory / "class_orientations.json"
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")

        volume_input = job.load_input("refinement_volume")
        rendering_output = job.alloc_output("rendering_map", volume_input)
        job.save_output("rendering_map", rendering_output)
        volume_path = _resolve_project_path(project, volume_input["map/path"][0])
        _, volume_data = mrc.read(volume_path)
        pixel_size = float(volume_input["map/psize_A"][0])
        select_particles = job.load_input("select_2d_particles")
        refinement_particles = job.load_input("refinement_particles")
        class_averages = _load_class_averages(
            project, job.load_input("select_2d_templates")
        )
        camera_results = {}
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
        for class_entry in artifact["classes"]:
            camera = camera_results[class_entry["class_id"]]
            axis = assign_symmetry_axis(
                camera.view_direction, symmetry, threshold_degrees=5
            )
            class_entry["camera"] = {
                "rotation_matrix": camera.rotation_matrix.tolist(),
                "quaternion_xyzw": camera.quaternion_xyzw.tolist(),
                "view_direction": camera.view_direction.tolist(),
                "in_plane_rotation_degrees": camera.in_plane_rotation_degrees,
                "projection_shift_pixels": camera.projection_shift_pixels.tolist(),
                "match_score": camera.match_score,
                "second_best_score": camera.second_best_score,
                "score_margin": camera.score_margin,
                "match_confidence": camera.match_confidence,
                "matching_box_size": int(
                    matching_grids[class_entry["class_id"]].class_average.shape[0]
                ),
                "matching_pixel_size_A": matching_grids[
                    class_entry["class_id"]
                ].pixel_size,
                "search_evaluation_count": camera.search_evaluation_count,
                "coordinate_convention": (
                    "right-handed Cartesian active rotation; "
                    "image rows increase downward"
                ),
            }
            class_entry["symmetry_axis"] = {
                "label": axis.label,
                "nearest_order": axis.nearest_order,
                "distance_degrees": axis.distance_degrees,
                "threshold_degrees": 5.0,
            }
        write_chimerax_bundle(
            job_directory / "chimerax",
            map_path=str(volume_path),
            cameras=camera_results,
        )
        output_path.write_text(json.dumps(artifact, indent=2) + "\n")
        projections = np.asarray(
            [camera_results[class_id].matched_projection for class_id in sorted(orientations)],
            dtype=np.float32,
        )
        projection_pixel_size = matching_grids[sorted(orientations)[0]].pixel_size
        mrc.write(
            job_directory / "class_projections.mrcs",
            projections,
            projection_pixel_size,
        )
        for class_number in interactive_class_numbers or ():
            class_id = class_number - 1
            if class_id not in camera_results:
                raise ValueError(f"Class {class_number} is not present in selected classes")
            name = f"class_{class_number:03d}_volume"
            rotated_volume = rotate_volume_at_rotation(
                volume_data, camera_results[class_id].rotation_matrix
            ).astype(np.float32, copy=False)
            filename = f"{name}.mrc"
            mrc.write(job_directory / filename, rotated_volume, pixel_size)
            volume_output = job.alloc_output(name, 1)
            volume_output["map/path"][:] = f">{job.uid}/{filename}"
            volume_output["map/shape"][:] = rotated_volume.shape
            volume_output["map/psize_A"][:] = pixel_size
            job.save_output(name, volume_output)
        preview = _class_result_preview(
            class_averages,
            projections,
            volume_data,
            camera_results,
            orientations,
        )
        projection_output = job.alloc_output("matched_projections", len(projections))
        projection_output["blob/path"][:] = f">{job.uid}/class_projections.mrcs"
        projection_output["blob/idx"][:] = np.arange(len(projections))
        projection_output["blob/shape"][:] = projections.shape[1:]
        projection_output["blob/psize_A"][:] = projection_pixel_size
        job.save_output("matched_projections", projection_output, image=preview)
        job.log_plot(preview, "Class projection preview", formats=["png"])
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


def _class_result_preview(
    class_averages, projections, volume, cameras, orientations, *, limit=25
):
    from matplotlib.figure import Figure

    class_ids = sorted(orientations)[:limit]
    figure = Figure(figsize=(9, 3 * len(class_ids)), constrained_layout=True)

    for row, class_id in enumerate(class_ids):
        orientation = orientations[class_id]
        class_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 1)
        class_axis.imshow(class_averages[class_id].image, cmap="gray")
        class_axis.set_title(
            f"Class {class_id + 1} | n={orientation.particle_count}\n"
            f"spread={orientation.angular_spread_degrees:.1f}°"
        )
        class_axis.axis("off")

        projection_axis = figure.add_subplot(len(class_ids), 3, row * 3 + 2)
        projection_axis.imshow(projections[row], cmap="gray")
        projection_axis.set_title(f"Matched | score={cameras[class_id].match_score:.3f}")
        projection_axis.axis("off")

        render_axis = figure.add_subplot(
            len(class_ids), 3, row * 3 + 3, projection="3d"
        )
        _plot_camera_render(render_axis, volume, cameras[class_id].rotation_matrix)
        render_axis.set_title("3D camera view")

    return figure


def _plot_camera_render(axis, volume, rotation_matrix, *, max_size=48):
    step = max(1, int(np.ceil(max(volume.shape) / max_size)))
    sampled = np.asarray(volume)[::step, ::step, ::step]
    positive = sampled[sampled > sampled.mean()]
    threshold = np.percentile(positive, 70) if positive.size else sampled.max()
    points_zyx = np.argwhere(sampled >= threshold).astype(float)
    if len(points_zyx) > 20000:
        points_zyx = points_zyx[:: int(np.ceil(len(points_zyx) / 20000))]
    center = (np.asarray(sampled.shape, dtype=float) - 1) / 2
    points_xyz = (points_zyx - center)[:, ::-1]
    rotated = points_xyz @ np.asarray(rotation_matrix).T
    if len(rotated):
        axis.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], s=2, alpha=0.35)
    radius = max(sampled.shape) / 2
    axis.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    axis.set_proj_type("ortho")
    axis.view_init(elev=90, azim=-90)
    axis.set_axis_off()
