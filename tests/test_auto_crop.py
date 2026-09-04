import numpy as np
import pytest

from cryosparc_2d_projection.auto_crop import (
    AutoCropDecision,
    PhysicalCameraView,
    compute_auto_crop_2d_framing,
    set_auto_crop_2d_limits,
)


def test_disabled_auto_crop_keeps_the_complete_native_frame():
    decision = compute_auto_crop_2d_framing(
        (32, 32),
        2.0,
        [],
        enabled=False,
    )

    assert decision.crop_bounds == (0, 0, 32, 32)
    assert decision.zoom == 1.0
    assert decision.enabled is False
    assert decision.fallback is False


def test_auto_crop_enabled_flag_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        compute_auto_crop_2d_framing(
            (32, 32),
            2.0,
            [],
            enabled=1,
        )


def test_even_crop_side_preserves_even_source_pixel_center():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        1.0,
        [PhysicalCameraView(camera_viewport_A=82.0)],
        enabled=True,
    )

    left, top, right, bottom = decision.crop_bounds
    assert ((left + right - 1) / 2.0, (top + bottom - 1) / 2.0) == (
        63.5,
        63.5,
    )


@pytest.mark.parametrize(
    ("source_size", "side", "expected_bounds", "expected_center"),
    [
        (127, 81, (23, 23, 104, 104), (63.0, 63.0)),
        (127, 82, (22, 22, 104, 104), (62.5, 62.5)),
        (128, 81, (24, 24, 105, 105), (64.0, 64.0)),
        (128, 82, (23, 23, 105, 105), (63.5, 63.5)),
    ],
)
def test_crop_pixel_center_rounding_is_deterministic_across_parities(
    source_size,
    side,
    expected_bounds,
    expected_center,
):
    decision = compute_auto_crop_2d_framing(
        (source_size, source_size),
        1.0,
        [PhysicalCameraView(camera_viewport_A=float(side))],
        enabled=True,
    )

    assert decision.crop_bounds == expected_bounds
    left, top, right, bottom = decision.crop_bounds
    selected_center = ((left + right - 1) / 2.0, (top + bottom - 1) / 2.0)
    native_center = ((source_size - 1) / 2.0,) * 2
    assert selected_center == expected_center
    assert max(
        abs(actual - target)
        for actual, target in zip(selected_center, native_center)
    ) <= 0.5


def test_projection_shift_uses_the_same_bounded_pixel_center_rounding():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        1.0,
        [
            PhysicalCameraView(
                camera_viewport_A=82.0,
                projection_shift_pixels=(1.25, -2.25),
            )
        ],
        enabled=True,
    )

    assert decision.crop_bounds == (24, 25, 106, 107)
    left, top, right, bottom = decision.crop_bounds
    selected_center = np.asarray(
        [(left + right - 1) / 2.0, (top + bottom - 1) / 2.0]
    )
    expected_center = np.asarray([64.75, 65.75])
    assert np.max(np.abs(selected_center - expected_center)) <= 0.5


def test_physical_camera_viewport_sets_native_crop_side_without_image_data():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [PhysicalCameraView(camera_viewport_A=80.0)],
        enabled=True,
    )

    assert decision.crop_shape == (40, 40)
    assert decision.crop_bounds == (44, 44, 84, 84)
    assert decision.zoom == pytest.approx(128 / 40)
    assert decision.fallback is False
    assert decision.clamped is False


def test_same_physical_viewport_scales_with_native_pixel_size():
    fine = compute_auto_crop_2d_framing(
        (128, 128),
        1.0,
        [PhysicalCameraView(camera_viewport_A=80.0)],
        enabled=True,
    )
    coarse = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [PhysicalCameraView(camera_viewport_A=80.0)],
        enabled=True,
    )

    assert fine.crop_shape == (80, 80)
    assert coarse.crop_shape == (40, 40)


def test_display_roll_does_not_change_square_camera_scale():
    unrolled = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [PhysicalCameraView(camera_viewport_A=80.0)],
        enabled=True,
    )
    rolled = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [PhysicalCameraView(camera_viewport_A=80.0, display_roll_degrees=90.0)],
        enabled=True,
    )

    assert rolled.crop_shape == unrolled.crop_shape
    assert rolled.zoom == unrolled.zoom


def test_projection_shift_is_flipped_for_display_then_rotated():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [
            PhysicalCameraView(
                camera_viewport_A=80.0,
                projection_shift_pixels=(6.0, 4.0),
                display_roll_degrees=90.0,
            )
        ],
        enabled=True,
    )

    # Raw (x=+6, y=+4) becomes display (x=+6, y=-4), then a 90-degree
    # counter-clockwise display roll gives (x=-4, y=-6).
    assert decision.crop_bounds == (40, 38, 80, 78)


def test_multiple_views_with_matching_centers_share_their_largest_scale():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [
            PhysicalCameraView(
                camera_viewport_A=70.0,
                projection_shift_pixels=(0, 0),
            ),
            PhysicalCameraView(
                camera_viewport_A=80.0,
                projection_shift_pixels=(0, 0),
            ),
        ],
        enabled=True,
    )

    assert decision.crop_shape == (40, 40)
    assert decision.crop_bounds == (44, 44, 84, 84)


def test_multiple_views_with_different_centers_use_range_midpoint():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [
            PhysicalCameraView(
                camera_viewport_A=80.0,
                projection_shift_pixels=(-8.0, 0.0),
            ),
            PhysicalCameraView(
                camera_viewport_A=80.0,
                projection_shift_pixels=(8.0, 0.0),
            ),
        ],
        enabled=True,
    )

    assert decision.fallback is False
    assert decision.crop_shape == (40, 40)
    assert decision.crop_bounds == (44, 44, 84, 84)


def test_multiple_view_centers_that_cannot_fit_one_crop_fall_back():
    decision = compute_auto_crop_2d_framing(
        (128, 128),
        2.0,
        [
            PhysicalCameraView(
                camera_viewport_A=40.0,
                projection_shift_pixels=(-30.0, 0.0),
            ),
            PhysicalCameraView(
                camera_viewport_A=40.0,
                projection_shift_pixels=(30.0, 0.0),
            ),
        ],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "view_center_range_exceeds_crop"


def test_native_source_shape_input_is_only_inspected_and_never_mutated():
    source_shape = np.array([128, 128], dtype=np.int64)
    before = source_shape.copy()

    compute_auto_crop_2d_framing(
        source_shape,
        2.0,
        [PhysicalCameraView(camera_viewport_A=80.0)],
        enabled=True,
    )

    assert np.array_equal(source_shape, before)


@pytest.mark.parametrize(
    ("view_framings", "reason"),
    [
        ([], "missing_view_framing"),
        ([object()], "invalid_view_framing"),
    ],
)
def test_invalid_or_missing_physical_view_falls_back_to_native_frame(
    view_framings, reason
):
    decision = compute_auto_crop_2d_framing(
        (64, 64),
        2.0,
        view_framings,
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == reason
    assert decision.crop_bounds == (0, 0, 64, 64)


def test_camera_viewport_larger_than_native_frame_falls_back():
    decision = compute_auto_crop_2d_framing(
        (64, 64),
        2.0,
        [PhysicalCameraView(camera_viewport_A=130.0)],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "camera_viewport_out_of_bounds"
    assert decision.crop_bounds == (0, 0, 64, 64)


def test_projection_shift_that_moves_crop_outside_native_frame_falls_back():
    decision = compute_auto_crop_2d_framing(
        (64, 64),
        2.0,
        [
            PhysicalCameraView(
                camera_viewport_A=40.0,
                projection_shift_pixels=(30.0, 0.0),
            )
        ],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "crop_out_of_bounds"
    assert decision.crop_bounds == (0, 0, 64, 64)


def test_view_center_outside_native_frame_falls_back_before_merging_views():
    decision = compute_auto_crop_2d_framing(
        (64, 64),
        2.0,
        [
            PhysicalCameraView(
                camera_viewport_A=40.0,
                projection_shift_pixels=(40.0, 0.0),
            )
        ],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "view_center_out_of_bounds"


@pytest.mark.parametrize("native_pixel_size_A", [0.0, np.nan, "not-a-number"])
def test_enabled_auto_crop_invalid_native_pixel_size_falls_back(
    native_pixel_size_A,
):
    decision = compute_auto_crop_2d_framing(
        (64, 64),
        native_pixel_size_A,
        [PhysicalCameraView(camera_viewport_A=40.0)],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "invalid_native_pixel_size"
    assert decision.crop_bounds == (0, 0, 64, 64)


def test_invalid_source_shape_is_rejected_before_enabled_fallback():
    view = [PhysicalCameraView(camera_viewport_A=40.0)]
    with pytest.raises(ValueError, match="square"):
        compute_auto_crop_2d_framing((64, 32), 0.0, view, enabled=True)


def test_disabled_auto_crop_still_rejects_invalid_native_pixel_size():
    with pytest.raises(ValueError, match="positive"):
        compute_auto_crop_2d_framing((64, 64), 0.0, [], enabled=False)


def test_physical_camera_view_normalizes_immutable_metadata():
    shift = np.array([2.0, -3.0], dtype=np.float32)
    view = PhysicalCameraView(
        camera_viewport_A=np.float32(40.0),
        projection_shift_pixels=shift,
    )
    shift[0] = 999.0

    assert view.camera_viewport_A == 40.0
    assert view.projection_shift_pixels == (2.0, -3.0)
    assert view.as_dict() == {
        "camera_viewport_A": 40.0,
        "projection_shift_pixels": [2.0, -3.0],
        "display_roll_degrees": 0.0,
    }


def test_set_auto_crop_2d_limits_changes_only_display_limits():
    decision = compute_auto_crop_2d_framing(
        (32, 32),
        1.0,
        [PhysicalCameraView(camera_viewport_A=16.0)],
        enabled=True,
    )

    class Axis:
        def __init__(self):
            self.xlim = None
            self.ylim = None

        def set_xlim(self, *values):
            self.xlim = values

        def set_ylim(self, *values):
            self.ylim = values

    axis = Axis()
    set_auto_crop_2d_limits(axis, decision)

    assert axis.xlim == (7.5, 23.5)
    assert axis.ylim == (23.5, 7.5)

    with pytest.raises(TypeError, match="AutoCropDecision"):
        set_auto_crop_2d_limits(axis, object())


def test_decision_keeps_legacy_metadata_slots_empty_for_physical_framing():
    decision = compute_auto_crop_2d_framing(
        (32, 32),
        1.0,
        [PhysicalCameraView(camera_viewport_A=16.0)],
        enabled=True,
    )

    assert isinstance(decision, AutoCropDecision)
    assert decision.foreground_bounds is None
    assert decision.silhouette_bounds is None
    assert decision.as_dict()["foreground_bounds"] is None
    assert decision.as_dict()["silhouette_bounds"] is None
