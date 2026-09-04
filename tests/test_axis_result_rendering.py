from dataclasses import replace
import json

import numpy as np
import pytest
from cryosparc import mrc

from cryosparc_2d_projection.axis_registry import get_axis_family
import cryosparc_2d_projection.axis_result_rendering as axis_result_rendering_module
from cryosparc_2d_projection.auto_crop import (
    PhysicalCameraView,
    compute_auto_crop_2d_framing,
)
from cryosparc_2d_projection.axis_result_rendering import (
    AxisResultRenderingError,
    AxisResultRenderingEventCode,
    AxisResultRenderingRequest,
    render_axis_search_results,
)
from cryosparc_2d_projection.axis_search import (
    AxisCandidate,
    AxisFamilyRanking,
    AxisProximityConfig,
    AxisRefinedCandidate,
    AxisRefinementResult,
    AxisSearchConfig,
    AxisSearchResult,
)
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.projection import project_volume_at_rotation
from cryosparc_2d_projection.surface_render import (
    build_surface_model,
    ClassRenderOptions,
    get_surface_camera_viewport_A,
    resolve_surface_sampling_grid,
)


def test_result_rendering_writes_one_complete_axis_search_result_set(tmp_path):
    request = _request(tmp_path)

    result = render_axis_search_results(request)

    artifact = json.loads((tmp_path / "axis_search_results.json").read_text())
    assert result.artifact == artifact
    assert artifact["symmetry"] == "I"
    assert artifact["families"] == ["2fold"]
    assert artifact["rows"][0]["class_number"] == 7
    assert artifact["rows"][0]["score_provenance"]
    assert artifact["rows"][0]["search_box_size"] == 16
    assert artifact["rows"][0]["search_pixel_size_A"] == 1.0
    assert artifact["rows"][0]["native_box_size"] == 16
    assert artifact["rows"][0]["native_pixel_size_A"] == 1.0
    assert artifact["rows"][0]["search_shift_xy_pixels"] == [0.0, 0.0]
    assert artifact["rows"][0]["search_evaluation_count"] == 1
    assert artifact["rows"][0]["refined_score"] is None
    assert artifact["rows"][0]["exact_axis_rotation_matrix"]
    assert artifact["rows"][0]["near_axis_rotation_matrix"] is None
    assert artifact["presentation"]["columns"] == [
        "Class Average",
        "Exact-Axis Matched Projection",
        "Exact-Axis Camera View",
    ]
    assert artifact["presentation"]["static_only"] is True
    assert set(result.stacks) == {
        "axis_candidates_raw",
        "axis_candidates_aligned",
        "axis_exact_references",
        "axis_exact_search_projections",
        "axis_exact_matched_projections",
        "axis_search_preview",
    }
    assert (tmp_path / "axis_exact_matched_projections.mrcs").is_file()
    assert (tmp_path / "axis_search_preview_001.png").is_file()
    assert len(result.preview_pages) == 1
    assert len(result.preview_pages[0].axes) == 3


def test_result_rendering_failure_names_candidate_and_leaves_no_partial_result(
    tmp_path,
):
    request = _request(tmp_path, invalid_second_candidate=True)

    with pytest.raises(Exception, match=r"family=2fold class=8"):
        render_axis_search_results(request)

    assert not (tmp_path / "axis_search_results.json").exists()
    assert not tuple(tmp_path.glob("*.mrcs"))
    assert not tuple(tmp_path.glob("*.png"))


def test_result_rendering_rejects_inconsistent_native_box_sizes(tmp_path):
    request = _request(tmp_path)
    family = get_axis_family("I", "2fold")
    smaller_volume = np.zeros((12, 12, 12), dtype=np.float32)
    smaller_volume[2:5, 3:7, 6:9] = 1.0
    smaller_projection = project_volume_at_rotation(
        smaller_volume,
        family.canonical_camera_matrix,
    )
    second = _candidate(family, smaller_projection, class_number=8)
    first = request.search_result.rows[0]
    search_result = AxisSearchResult(
        families={
            family.name: replace(
                request.search_result.families[family.name],
                candidates=(first, second),
            )
        },
        rows=(first, second),
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=8 output=axis_candidates_raw.*16.*12",
    ):
        render_axis_search_results(replace(request, search_result=search_result))

    assert not (tmp_path / "axis_search_results.json").exists()


def test_result_rendering_ignores_progress_receiver_failure(tmp_path):
    def fail(_event):
        raise RuntimeError("receiver unavailable")

    request = replace(_request(tmp_path), progress_callback=fail)

    result = render_axis_search_results(request)

    assert result.artifact["rows"][0]["class_number"] == 7
    assert (tmp_path / "axis_search_results.json").is_file()


def test_result_rendering_progress_uses_stable_event_codes(tmp_path):
    events = []
    request = replace(_request(tmp_path), progress_callback=events.append)

    render_axis_search_results(request)

    assert {event.code for event in events} >= {
        AxisResultRenderingEventCode.SURFACE_SAMPLING,
        AxisResultRenderingEventCode.CANDIDATE_COMPLETED,
    }


def test_result_rendering_output_error_names_candidate_and_output(
    tmp_path,
    monkeypatch,
):
    def fail_write(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        "cryosparc_2d_projection.axis_result_rendering.mrc.write",
        fail_write,
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=axis_candidates_raw",
    ):
        render_axis_search_results(_request(tmp_path))


def test_native_reprojection_error_names_candidate_and_output(tmp_path):
    request = replace(
        _request(tmp_path),
        matching_map=np.zeros((16, 16), dtype=np.float32),
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=(
            r"family=2fold class=7 "
            r"output=axis_exact_matched_projections"
        ),
    ):
        render_axis_search_results(request)


def test_reference_projection_error_names_candidate_and_output(
    tmp_path,
    monkeypatch,
):
    def fail_projection(*_args, **_kwargs):
        raise OSError("projection unavailable")

    monkeypatch.setattr(
        "cryosparc_2d_projection.axis_result_rendering.project_volume_at_rotation",
        fail_projection,
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=axis_exact_references",
    ):
        render_axis_search_results(_request(tmp_path))


def test_native_diagnostic_error_names_candidate_and_output(
    tmp_path,
    monkeypatch,
):
    def fail_diagnostic(*_args, **_kwargs):
        raise OSError("diagnostic unavailable")

    monkeypatch.setattr(
        "cryosparc_2d_projection.axis_result_rendering."
        "compute_diagnostic_band_limited_score",
        fail_diagnostic,
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=axis_search_results.json",
    ):
        render_axis_search_results(_request(tmp_path))


def test_camera_view_render_error_names_candidate_and_output(
    tmp_path,
    monkeypatch,
):
    def fail_render(*_args, **_kwargs):
        raise OSError("render unavailable")

    monkeypatch.setattr(
        "cryosparc_2d_projection.axis_result_rendering.write_camera_view_render",
        fail_render,
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=(
            r"family=2fold class=7 "
            r"output=renders/exact/class_001_exact.png"
        ),
    ):
        render_axis_search_results(_request(tmp_path))


def test_result_rendering_can_replace_one_complete_result_set(tmp_path):
    first = render_axis_search_results(_request(tmp_path))

    second = render_axis_search_results(_request(tmp_path))

    assert second.artifact["rows"] == first.artifact["rows"]
    assert (tmp_path / "axis_search_results.json").is_file()
    assert (tmp_path / "axis_search_preview_001.png").is_file()


def test_failed_replacement_preserves_previous_complete_result_set(tmp_path):
    render_axis_search_results(_request(tmp_path))
    previous_json = (tmp_path / "axis_search_results.json").read_bytes()

    with pytest.raises(AxisResultRenderingError):
        render_axis_search_results(
            _request(tmp_path, invalid_second_candidate=True)
        )

    assert (tmp_path / "axis_search_results.json").read_bytes() == previous_json
    assert (tmp_path / "axis_exact_matched_projections.mrcs").is_file()


def test_failed_delivery_does_not_report_output_writing_completed(
    tmp_path,
    monkeypatch,
):
    events = []

    def fail_delivery(*_args, **_kwargs):
        raise OSError("delivery unavailable")

    monkeypatch.setattr(
        "cryosparc_2d_projection.axis_result_rendering._promote_complete_result",
        fail_delivery,
    )
    request = replace(_request(tmp_path), progress_callback=events.append)

    with pytest.raises(AxisResultRenderingError):
        render_axis_search_results(request)

    assert AxisResultRenderingEventCode.OUTPUT_WRITING_STARTED in {
        event.code for event in events
    }
    assert AxisResultRenderingEventCode.OUTPUT_WRITING_COMPLETED not in {
        event.code for event in events
    }


def test_result_rendering_keeps_near_axis_and_native_outputs_consistent(tmp_path):
    request = _near_axis_request(
        tmp_path,
        native_size=18,
        search_size=16,
    )

    result = render_axis_search_results(request)

    assert {
        "axis_near_projections",
        "axis_near_search_projections",
        "axis_near_matched_projections",
    } <= set(result.stacks)
    assert result.artifact["rows"][0]["refined_score"] == 0.95
    assert result.artifact["rows"][0]["angular_distance_degrees"] == 1.25
    assert result.artifact["presentation"]["columns"] == [
        "Class Average",
        "Near-Axis Matched Projection",
        "Exact-Axis Matched Projection",
        "Near-Axis Camera View",
        "Exact-Axis Camera View",
    ]
    _, search = mrc.read(tmp_path / "axis_exact_search_projections.mrcs")
    _, matched = mrc.read(tmp_path / "axis_exact_matched_projections.mrcs")
    assert search.shape[1:] == (16, 16)
    assert matched.shape[1:] == (18, 18)
    assert len(result.preview_pages[0].axes) == 5


def test_result_rendering_rejects_near_axis_mode_without_refinement(tmp_path):
    request = replace(_request(tmp_path), refine_near_axis=True)

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=near-axis-result",
    ):
        render_axis_search_results(request)

    assert not (tmp_path / "axis_search_results.json").exists()


def test_result_rendering_rejects_refinement_when_near_axis_mode_is_disabled(
    tmp_path,
):
    near_request = _near_axis_request(tmp_path)
    request = replace(near_request, refine_near_axis=False)

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=near-axis-result",
    ):
        render_axis_search_results(request)

    assert not (tmp_path / "axis_search_results.json").exists()


def test_result_rendering_rejects_extra_refinement_rows(tmp_path):
    request = _near_axis_request(tmp_path)
    refined = request.refinement.rows[0]
    request = replace(
        request,
        refinement=replace(request.refinement, rows=(refined, refined)),
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=near-axis-result",
    ):
        render_axis_search_results(request)

    assert not (tmp_path / "axis_search_results.json").exists()


def test_result_rendering_rejects_refinement_for_a_different_candidate(tmp_path):
    request = _near_axis_request(tmp_path)
    refined = request.refinement.rows[0]
    wrong_candidate = replace(refined.exact_candidate, class_number=8)
    request = replace(
        request,
        refinement=replace(
            request.refinement,
            rows=(replace(refined, exact_candidate=wrong_candidate),),
        ),
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=near-axis-result",
    ):
        render_axis_search_results(request)


def test_replacement_removes_outputs_not_present_in_new_result_set(tmp_path):
    render_axis_search_results(_near_axis_request(tmp_path))
    assert tuple(tmp_path.glob("axis_near_*.mrcs"))

    render_axis_search_results(_request(tmp_path))

    assert not tuple(tmp_path.glob("axis_near_*.mrcs"))


def test_candidate_assembly_error_names_actual_candidate_and_output(tmp_path):
    request = _request(tmp_path)
    request = replace(
        request,
        refinement=AxisRefinementResult(
            exact_result=request.search_result,
            rows=(),
        ),
        refine_near_axis=True,
    )

    with pytest.raises(
        AxisResultRenderingError,
        match=r"family=2fold class=7 output=near-axis-result",
    ):
        render_axis_search_results(request)


def test_result_rendering_presentation_options_do_not_change_scientific_results(
    tmp_path,
):
    first_request = _request(tmp_path / "first")
    second_request = replace(
        _request(tmp_path / "second"),
        axis_rolls={"2fold": 25.0},
        comparison_options=ComparisonRenderOptions(dpi=150, page_size=1),
    )

    first = render_axis_search_results(first_request)
    second = render_axis_search_results(second_request)

    scientific_keys = (
        "axis_class_score",
        "refined_score",
        "roll_degrees",
        "shift_xy_pixels",
        "near_axis_rotation_matrix",
        "angular_distance_degrees",
    )
    assert {
        key: first.artifact["rows"][0][key] for key in scientific_keys
    } == {
        key: second.artifact["rows"][0][key] for key in scientific_keys
    }
    assert np.array_equal(
        first.stacks["axis_candidates_raw"].data,
        second.stacks["axis_candidates_raw"].data,
    )
    assert not np.array_equal(
        first.stacks["axis_candidates_aligned"].data,
        second.stacks["axis_candidates_aligned"].data,
    )


def test_result_rendering_auto_crops_exact_axis_2d_panels(tmp_path):
    request = replace(
        _request(tmp_path, native_size=64),
        comparison_options=ComparisonRenderOptions(
            page_size=1,
            auto_crop_2d=True,
        ),
        rendering_pixel_size_A=0.5,
    )

    result = render_axis_search_results(request)

    framing = result.artifact["rows"][0]["presentation"]["auto_crop_2d"]
    assert result.artifact["presentation"]["auto_crop_2d"]["enabled"] is True
    assert framing["fallback"] is False
    assert framing["zoom"] > 1.0
    crop = framing["crop_bounds"]
    expected_xlim = (crop["left"] - 0.5, crop["right"] - 0.5)
    expected_ylim = (crop["bottom"] - 0.5, crop["top"] - 0.5)
    axes = result.preview_pages[0].axes
    for axis in axes[:2]:
        assert axis.images[0].get_array().shape == (64, 64)
        assert np.allclose(axis.get_xlim(), expected_xlim)
        assert np.allclose(axis.get_ylim(), expected_ylim)
    assert axes[2].images[0].get_array().shape == (64, 64, 3)


def test_near_axis_auto_crop_uses_rolled_display_projections(tmp_path):
    request = replace(
        _near_axis_request(tmp_path, native_size=64),
        axis_rolls={"2fold": 90.0},
        comparison_options=ComparisonRenderOptions(
            page_size=1,
            auto_crop_2d=True,
        ),
    )

    result = render_axis_search_results(request)

    framing = result.artifact["rows"][0]["presentation"]["auto_crop_2d"]
    sampling_grid = resolve_surface_sampling_grid(
        request.rendering_map.shape,
        request.render_options.grid_size,
    )
    surface = build_surface_model(
        request.rendering_map,
        surface_level=request.render_options.surface_level,
        sampling_grid=sampling_grid,
    )
    camera_viewport_A = get_surface_camera_viewport_A(
        surface,
        rendering_pixel_size_A=request.rendering_pixel_size_A,
    )
    native_shift = tuple(result.artifact["rows"][0]["native_shift_xy_pixels"])
    expected = compute_auto_crop_2d_framing(
        result.stacks["axis_exact_matched_projections"].data[0].shape,
        result.stacks["axis_exact_matched_projections"].pixel_size_A,
        [
            PhysicalCameraView(
                camera_viewport_A=camera_viewport_A,
                projection_shift_pixels=native_shift,
                display_roll_degrees=90.0,
            ),
            PhysicalCameraView(
                camera_viewport_A=camera_viewport_A,
                projection_shift_pixels=native_shift,
                display_roll_degrees=90.0,
            ),
        ],
        enabled=True,
    )
    assert framing == expected.as_dict()
    crop = framing["crop_bounds"]
    expected_xlim = (crop["left"] - 0.5, crop["right"] - 0.5)
    expected_ylim = (crop["bottom"] - 0.5, crop["top"] - 0.5)
    for axis in result.preview_pages[0].axes[:3]:
        assert np.allclose(axis.get_xlim(), expected_xlim)
        assert np.allclose(axis.get_ylim(), expected_ylim)


def test_axis_auto_crop_ignores_projection_contrast_and_noise(
    tmp_path,
    monkeypatch,
):
    clean = np.zeros((32, 32), dtype=np.float32)
    clean[10:22, 10:22] = 1.0
    noisy = clean.copy()
    noisy[1:5, 1:5] = 10.0
    current_projection = clean

    monkeypatch.setattr(
        axis_result_rendering_module,
        "get_surface_camera_viewport_A",
        lambda *_args, **_kwargs: 16.0,
    )

    def fake_native_match(
        request,
        candidate,
        rotation_matrix,
        *,
        reference_rotation_matrix,
        pass_name,
    ):
        return {
            "projection": current_projection,
            "reference_projection": current_projection,
            "matched_projection": current_projection,
            "shift_xy_pixels": [0.0, 0.0],
            "pixel_size_A": 1.0,
            "diagnostic_score": {"score": 0.9, "valid": True},
        }

    monkeypatch.setattr(
        axis_result_rendering_module,
        "_render_native_match",
        fake_native_match,
    )
    request = replace(
        _request(tmp_path / "clean", native_size=32),
        comparison_options=ComparisonRenderOptions(auto_crop_2d=True),
    )
    clean_result = render_axis_search_results(request)

    current_projection = noisy
    noisy_result = render_axis_search_results(
        replace(request, output_directory=tmp_path / "noisy")
    )

    assert (
        clean_result.artifact["rows"][0]["presentation"]["auto_crop_2d"]
        == noisy_result.artifact["rows"][0]["presentation"]["auto_crop_2d"]
    )


def test_axis_auto_crop_scales_with_rendering_pixel_size(
    tmp_path,
    monkeypatch,
):
    requested_viewports = []

    def fake_viewport(_surface, *, rendering_pixel_size_A):
        requested_viewports.append(rendering_pixel_size_A)
        return 24.0 * rendering_pixel_size_A

    monkeypatch.setattr(
        axis_result_rendering_module,
        "get_surface_camera_viewport_A",
        fake_viewport,
    )
    options = ComparisonRenderOptions(auto_crop_2d=True)
    fine = render_axis_search_results(
        replace(
            _request(tmp_path / "fine", native_size=64),
            comparison_options=options,
            rendering_pixel_size_A=1.0,
        )
    )
    coarse = render_axis_search_results(
        replace(
            _request(tmp_path / "coarse", native_size=64),
            comparison_options=options,
            rendering_pixel_size_A=2.0,
        )
    )

    assert requested_viewports == [1.0, 2.0]
    fine_shape = fine.artifact["rows"][0]["presentation"]["auto_crop_2d"][
        "crop_shape"
    ]
    coarse_shape = coarse.artifact["rows"][0]["presentation"]["auto_crop_2d"][
        "crop_shape"
    ]
    assert fine_shape == [24, 24]
    assert coarse_shape == [48, 48]


def test_near_axis_auto_crop_merges_exact_and_near_centers_at_range_midpoint(
    tmp_path,
    monkeypatch,
):
    request = replace(
        _near_axis_request(tmp_path, native_size=32),
        comparison_options=ComparisonRenderOptions(auto_crop_2d=True),
    )
    monkeypatch.setattr(
        axis_result_rendering_module,
        "get_surface_camera_viewport_A",
        lambda *_args, **_kwargs: 16.0,
    )

    def fake_native_match(
        request,
        candidate,
        rotation_matrix,
        *,
        reference_rotation_matrix,
        pass_name,
    ):
        shift = (6.0, 4.0) if pass_name == "exact" else (-6.0, -4.0)
        projection = np.zeros((32, 32), dtype=np.float32)
        projection[8:24, 8:24] = 1.0
        return {
            "projection": projection,
            "reference_projection": projection,
            "matched_projection": projection,
            "shift_xy_pixels": list(shift),
            "pixel_size_A": 1.0,
            "diagnostic_score": {"score": 0.9, "valid": True},
        }

    monkeypatch.setattr(
        axis_result_rendering_module,
        "_render_native_match",
        fake_native_match,
    )

    result = render_axis_search_results(request)

    framing = result.artifact["rows"][0]["presentation"]["auto_crop_2d"]
    assert framing["fallback"] is False
    assert framing["crop_bounds"] == {
        "left": 8,
        "top": 8,
        "right": 24,
        "bottom": 24,
    }


def test_axis_auto_crop_invalid_camera_viewport_falls_back_with_warning(
    tmp_path,
    monkeypatch,
):
    def fail_viewport(*_args, **_kwargs):
        raise ValueError("invalid physical viewport")

    monkeypatch.setattr(
        axis_result_rendering_module,
        "get_surface_camera_viewport_A",
        fail_viewport,
    )
    result = render_axis_search_results(
        replace(
            _request(tmp_path, native_size=32),
            comparison_options=ComparisonRenderOptions(auto_crop_2d=True),
        )
    )

    framing = result.artifact["rows"][0]["presentation"]["auto_crop_2d"]
    assert framing["fallback"] is True
    assert framing["crop_shape"] == [32, 32]
    assert any(
        warning.code is AxisResultRenderingEventCode.WARNING
        and "invalid physical camera viewport" in warning.message
        for warning in result.warnings
    )


def test_axis_auto_crop_preserves_scientific_rows_and_stacks(tmp_path):
    disabled = render_axis_search_results(
        replace(
            _request(tmp_path / "disabled", native_size=64),
            comparison_options=ComparisonRenderOptions(auto_crop_2d=False),
        )
    )
    enabled = render_axis_search_results(
        replace(
            _request(tmp_path / "enabled", native_size=64),
            comparison_options=ComparisonRenderOptions(auto_crop_2d=True),
        )
    )

    assert {
        key: value
        for key, value in enabled.artifact["rows"][0].items()
        if key != "presentation"
    } == disabled.artifact["rows"][0]
    for name in disabled.stacks:
        assert np.array_equal(enabled.stacks[name].data, disabled.stacks[name].data)
        assert enabled.stacks[name].pixel_size_A == disabled.stacks[name].pixel_size_A


def _request(
    tmp_path,
    *,
    invalid_second_candidate=False,
    native_size=16,
    search_size=None,
):
    size = native_size
    volume = np.zeros((size, size, size), dtype=np.float32)
    volume[
        int(0.22 * size) : int(0.38 * size),
        int(0.28 * size) : int(0.47 * size),
        int(0.56 * size) : int(0.75 * size),
    ] = 1.0
    volume[
        int(0.62 * size) : int(0.78 * size),
        int(0.53 * size) : int(0.72 * size),
        int(0.19 * size) : int(0.31 * size),
    ] = 0.6
    family = get_axis_family("I", "2fold")
    projection = project_volume_at_rotation(volume, family.canonical_camera_matrix)
    search_projection = projection
    if search_size is not None:
        start = (native_size - search_size) // 2
        search_projection = projection[
            start : start + search_size,
            start : start + search_size,
        ]
    candidate = _candidate(
        family,
        projection,
        class_number=7,
        search_projection=search_projection,
    )
    candidates = [candidate]
    if invalid_second_candidate:
        candidates.append(_candidate(family, projection[:, 0], class_number=8))
    search_result = AxisSearchResult(
        families={
            family.name: AxisFamilyRanking(
                family=family,
                candidates=tuple(candidates),
                first_score=0.9,
            )
        },
        rows=tuple(candidates),
    )
    return AxisResultRenderingRequest(
        output_directory=tmp_path,
        search_result=search_result,
        refinement=None,
        matching_map=volume,
        rendering_map=volume,
        class_pixel_size_A=1.0,
        map_pixel_size_A=1.0,
        rendering_pixel_size_A=1.0,
        config=AxisSearchConfig(top_n=2),
        proximity_config=AxisProximityConfig(),
        axis_rolls={},
        comparison_options=ComparisonRenderOptions(page_size=2),
        render_options=ClassRenderOptions(image_size=64, grid_size=size),
        refine_near_axis=False,
        timings={},
    )


def _candidate(family, projection, *, class_number, search_projection=None):
    search_projection = projection if search_projection is None else search_projection
    return AxisCandidate(
        family_name=family.name,
        class_number=class_number,
        raw_class=projection,
        aligned_class=projection,
        exact_reference=projection,
        exact_reference_display=np.flipud(projection),
        search_projection=search_projection,
        canonical_axis_rotation_matrix=family.canonical_camera_matrix,
        exact_rotation_matrix=family.canonical_camera_matrix,
        exact_score=0.9,
        raw_correlation=0.9,
        roll_degrees=0.0,
        shift_xy_pixels=(0.0, 0.0),
        score_metadata={
            "search_box_size": search_projection.shape[0],
            "search_pixel_size_A": 1.0,
            "search_evaluation_count": 1,
        },
    )


def _near_axis_request(tmp_path, *, native_size=16, search_size=None):
    request = _request(
        tmp_path,
        native_size=native_size,
        search_size=search_size,
    )
    candidate = request.search_result.rows[0]
    refined = AxisRefinedCandidate(
        exact_candidate=candidate,
        refined_score=0.95,
        angular_distance_degrees=1.25,
        canonical_near_axis_rotation_matrix=(
            candidate.canonical_axis_rotation_matrix
        ),
        near_axis_rotation_matrix=candidate.exact_rotation_matrix,
        near_axis_projection=candidate.search_projection,
        near_axis_projection_display=np.flipud(candidate.search_projection),
        matched_search_projection=candidate.search_projection,
        roll_degrees=0.0,
        shift_xy_pixels=(0.0, 0.0),
        cone_boundary=False,
    )
    return replace(
        request,
        refinement=AxisRefinementResult(
            exact_result=request.search_result,
            rows=(refined,),
        ),
        refine_near_axis=True,
    )
