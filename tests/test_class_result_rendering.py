from dataclasses import replace
from pathlib import Path

import numpy as np
from cryosparc import mrc
import pytest

import cryosparc_2d_projection.class_result_rendering as class_result_rendering_module
from cryosparc_2d_projection.camera import ClassCameraResult
from cryosparc_2d_projection.class_poses import ClassOrientation
from cryosparc_2d_projection.class_result_rendering import (
    ClassResultInput,
    ClassResultRenderingError,
    ClassResultRenderingRequest,
    NativeReprojectionError,
    render_class_results,
)
from cryosparc_2d_projection.external_job_adapter import LoadedClassAverage
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.surface_render import (
    ClassRenderOptions,
    SurfaceRenderMemoryError,
    resolve_surface_sampling_grid,
)


def _one_class_request(tmp_path, *, auto_crop=False):
    matching_map = np.zeros((7, 7, 7), dtype=np.float32)
    matching_map[1:6, 1:6, 1:6] = 1.0
    matching_map[1, 5, 3] = 3.0
    class_average = matching_map.sum(axis=0)
    camera = ClassCameraResult(
        rotation_matrix=np.eye(3),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        view_direction=np.array([0.0, 0.0, 1.0]),
        in_plane_rotation_degrees=0.0,
        matched_projection=class_average.copy(),
        projection_shift_pixels=np.zeros(2),
        match_score=1.0,
        second_best_score=0.5,
        score_margin=0.5,
        match_confidence="high",
        search_evaluation_count=1,
    )
    return ClassResultRenderingRequest(
        output_directory=tmp_path,
        classes=(
            ClassResultInput(
                class_id=0,
                class_average=LoadedClassAverage(
                    image=class_average,
                    pixel_size_A=1.5,
                    source_index=0,
                    source_path=tmp_path / "templates.mrcs",
                ),
                orientation=ClassOrientation(
                    particle_count=10,
                    view_direction=np.array([0.0, 0.0, 1.0]),
                    angular_spread_degrees=2.0,
                ),
                camera=camera,
                search_projection=class_average.copy(),
                search_pixel_size_A=1.5,
            ),
        ),
        matching_map=matching_map,
        matching_pixel_size_A=1.5,
        rendering_map=matching_map,
        rendering_pixel_size_A=1.5,
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=7,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(
            dpi=40,
            page_size=1,
            auto_crop_2d=auto_crop,
        ),
    )


def test_class_result_rendering_rejects_an_empty_result_set(tmp_path):
    request = ClassResultRenderingRequest(
        output_directory=tmp_path,
        classes=(),
        matching_map=np.zeros((5, 5, 5), dtype=np.float32),
        matching_pixel_size_A=1.5,
        rendering_map=np.zeros((5, 5, 5), dtype=np.float32),
        rendering_pixel_size_A=1.5,
        symmetry="C1",
    )

    with pytest.raises(
        ClassResultRenderingError,
        match="at least one Class Result",
    ):
        render_class_results(request)

    assert tuple(tmp_path.iterdir()) == ()


def test_class_result_rendering_writes_one_complete_local_result_set(tmp_path):
    matching_map = np.zeros((7, 7, 7), dtype=np.float32)
    matching_map[1:6, 1:6, 1:6] = 1.0
    matching_map[1, 5, 3] = 3.0
    class_average = matching_map.sum(axis=0)
    camera = ClassCameraResult(
        rotation_matrix=np.eye(3),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        view_direction=np.array([0.0, 0.0, 1.0]),
        in_plane_rotation_degrees=0.0,
        matched_projection=class_average.copy(),
        projection_shift_pixels=np.zeros(2),
        match_score=1.0,
        second_best_score=0.5,
        score_margin=0.5,
        match_confidence="high",
        search_evaluation_count=1,
    )
    request = ClassResultRenderingRequest(
        output_directory=tmp_path,
        classes=(
            ClassResultInput(
                class_id=0,
                class_average=LoadedClassAverage(
                    image=class_average,
                    pixel_size_A=1.5,
                    source_index=0,
                    source_path=tmp_path / "templates.mrcs",
                ),
                orientation=ClassOrientation(
                    particle_count=10,
                    view_direction=np.array([0.0, 0.0, 1.0]),
                    angular_spread_degrees=2.0,
                ),
                camera=camera,
                search_projection=class_average.copy(),
                search_pixel_size_A=1.5,
            ),
        ),
        matching_map=matching_map,
        matching_pixel_size_A=1.5,
        rendering_map=matching_map,
        rendering_pixel_size_A=1.5,
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=7,
            surface_level=0.5,
        ),
        comparison_options=ComparisonRenderOptions(dpi=40, page_size=1),
    )

    result = render_class_results(request)

    assert result.reproducibility_metadata_path == (
        tmp_path / "class_orientations.json"
    )
    assert result.reproducibility_metadata["classes"][0]["class_number"] == 1
    assert set(result.stacks) == {"matched_projections", "search_projections"}
    _, matched = mrc.read(result.stacks["matched_projections"].path)
    _, searched = mrc.read(result.stacks["search_projections"].path)
    assert matched.shape == searched.shape == (1, 7, 7)
    assert result.render_paths[0].exists()
    assert result.comparison_paths[0].exists()
    assert result.thumbnail_path.exists()
    assert len(result.preview_paths) == 1
    assert result.preview_paths[0].exists()
    assert not hasattr(result, "preview_pages")


def test_class_result_rendering_preserves_the_previous_set_on_failure(tmp_path):
    old_metadata = tmp_path / "class_orientations.json"
    old_metadata.write_text("old complete result\n")
    old_render = tmp_path / "renders" / "old.png"
    old_render.parent.mkdir()
    old_render.write_bytes(b"old render")
    matching_map = np.zeros((5, 5, 5), dtype=np.float32)
    matching_map[1:4, 1:4, 1:4] = 1.0
    class_average = matching_map.sum(axis=0)
    request = ClassResultRenderingRequest(
        output_directory=tmp_path,
        classes=(
            ClassResultInput(
                class_id=0,
                class_average=LoadedClassAverage(
                    image=class_average,
                    pixel_size_A=1.5,
                    source_index=0,
                    source_path=tmp_path / "templates.mrcs",
                ),
                orientation=ClassOrientation(
                    particle_count=1,
                    view_direction=np.array([0.0, 0.0, 1.0]),
                    angular_spread_degrees=0.0,
                ),
                camera=ClassCameraResult(
                    rotation_matrix=np.eye(3),
                    quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                    view_direction=np.array([0.0, 0.0, 1.0]),
                    in_plane_rotation_degrees=0.0,
                    matched_projection=class_average.copy(),
                    projection_shift_pixels=np.zeros(2),
                    match_score=1.0,
                ),
                search_projection=class_average.copy(),
                search_pixel_size_A=1.5,
            ),
        ),
        matching_map=matching_map,
        matching_pixel_size_A=1.5,
        rendering_map=np.zeros((5, 5, 5), dtype=np.float32),
        rendering_pixel_size_A=1.5,
        symmetry="C1",
        render_options=ClassRenderOptions(
            image_size=64,
            grid_size=5,
            surface_level=0.5,
        ),
    )

    with pytest.raises(ClassResultRenderingError, match="surface"):
        render_class_results(request)

    assert old_metadata.read_text() == "old complete result\n"
    assert old_render.read_bytes() == b"old render"
    assert not any(path.name.startswith(".class-result-") for path in tmp_path.iterdir())


def test_class_result_rendering_restores_every_previous_path_on_promotion_failure(
    tmp_path,
    monkeypatch,
):
    previous = {
        "class_orientations.json": b"old metadata",
        "class_projections.mrcs": b"old matched",
        "search_projections.mrcs": b"old search",
    }
    for name, content in previous.items():
        (tmp_path / name).write_bytes(content)
    old_render = tmp_path / "renders" / "old.png"
    old_render.parent.mkdir()
    old_render.write_bytes(b"old render")
    real_replace = Path.replace
    failure_injected = False

    def fail_during_new_stack_promotion(source, target):
        nonlocal failure_injected
        target = Path(target)
        if (
            not failure_injected
            and source.name == "class_projections.mrcs"
            and source.parent.name.startswith(".class-result-staging-")
            and target.parent == tmp_path
        ):
            failure_injected = True
            raise OSError("injected promotion failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_during_new_stack_promotion)

    with pytest.raises(ClassResultRenderingError, match="promotion failed"):
        render_class_results(_one_class_request(tmp_path))

    assert failure_injected is True
    for name, content in previous.items():
        assert (tmp_path / name).read_bytes() == content
    assert old_render.read_bytes() == b"old render"
    assert not any(path.name.startswith(".class-result-") for path in tmp_path.iterdir())


def test_class_result_rendering_callback_failures_do_not_invalidate_results(tmp_path):
    def fail_callback(event):
        raise RuntimeError(f"observer failed during {event.stage}")

    request = replace(
        _one_class_request(tmp_path),
        comparison_options=ComparisonRenderOptions(dpi=101, page_size=1),
        progress_callback=fail_callback,
        warning_callback=fail_callback,
    )

    result = render_class_results(request)

    assert result.reproducibility_metadata_path.exists()
    assert result.warnings


def test_class_result_rendering_reports_invalid_camera_viewport_once(
    tmp_path,
    monkeypatch,
):
    def fail_camera_viewport(*args, **kwargs):
        raise ValueError("invalid surface units")

    monkeypatch.setattr(
        class_result_rendering_module,
        "get_surface_camera_viewport_A",
        fail_camera_viewport,
    )
    warnings = []
    request = replace(
        _one_class_request(tmp_path, auto_crop=True),
        warning_callback=warnings.append,
    )

    result = render_class_results(request)

    fallback_warnings = [
        event for event in warnings if "Auto-Cropped 2D Framing fell back" in event.message
    ]
    assert len(fallback_warnings) == 1
    assert "invalid surface units" in fallback_warnings[0].message
    framing = result.reproducibility_metadata["classes"][0]["presentation"][
        "auto_crop_2d"
    ]
    assert framing["fallback"] is True


def test_class_result_rendering_names_native_reprojection_failure(
    tmp_path,
    monkeypatch,
):
    def fail_native_reprojection(*args, **kwargs):
        raise MemoryError("injected native reprojection failure")

    monkeypatch.setattr(
        class_result_rendering_module,
        "project_native_matched_projection",
        fail_native_reprojection,
    )

    with pytest.raises(NativeReprojectionError) as raised:
        render_class_results(_one_class_request(tmp_path))

    assert "Class 1" in str(raised.value)
    assert "bounded Search Projection was not substituted" in str(raised.value)
    assert tuple(tmp_path.iterdir()) == ()


def test_class_result_rendering_preserves_surface_memory_failure(
    tmp_path,
    monkeypatch,
):
    failure = SurfaceRenderMemoryError(
        stage="surface extraction",
        sampling_grid=resolve_surface_sampling_grid(
            (512, 384, 256),
            requested_grid_size=512,
        ),
    )

    def fail_surface_build(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        class_result_rendering_module,
        "build_surface_model",
        fail_surface_build,
    )

    with pytest.raises(SurfaceRenderMemoryError) as raised:
        render_class_results(_one_class_request(tmp_path))

    assert raised.value is failure
    assert "--render-grid-size 384" in str(raised.value)


def test_class_result_rendering_reports_sampling_before_surface_extraction(
    tmp_path,
    monkeypatch,
):
    events = []
    real_build_surface_model = class_result_rendering_module.build_surface_model

    def assert_sampling_was_reported(*args, **kwargs):
        assert any(
            event.stage == "surface-sampling"
            and event.message.startswith("Surface Sampling Grid:")
            for event in events
        )
        return real_build_surface_model(*args, **kwargs)

    monkeypatch.setattr(
        class_result_rendering_module,
        "build_surface_model",
        assert_sampling_was_reported,
    )
    request = replace(_one_class_request(tmp_path), progress_callback=events.append)

    render_class_results(request)
