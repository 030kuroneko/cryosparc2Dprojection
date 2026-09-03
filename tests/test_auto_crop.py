import numpy as np
import pytest

from cryosparc_2d_projection.auto_crop import (
    compute_auto_crop_2d_framing,
)
from cryosparc_2d_projection.surface_render import SurfaceSilhouetteBounds


_HALF_FRAME_SILHOUETTE = SurfaceSilhouetteBounds(0.25, 0.25, 0.75, 0.75)


def test_disabled_auto_crop_keeps_the_complete_native_frame():
    matched_projection = np.zeros((32, 32), dtype=np.float32)

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [_HALF_FRAME_SILHOUETTE],
        enabled=False,
    )

    assert decision.crop_bounds == (0, 0, 32, 32)
    assert decision.zoom == 1.0
    assert decision.enabled is False
    assert decision.fallback is False


def test_auto_crop_frames_native_2d_pair_to_surface_occupancy():
    matched_projection = np.zeros((32, 32), dtype=np.float32)
    matched_projection[12:20, 13:21] = 1.0
    class_average = matched_projection.copy()
    silhouette = SurfaceSilhouetteBounds(
        left=0.25,
        top=0.25,
        right=0.75,
        bottom=0.75,
    )

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [silhouette],
        enabled=True,
    )

    # Matching the raw 8 px foreground to a 50% silhouette needs a 16 px
    # display window, which contains its 2 px padding on every side.
    assert decision.crop_bounds == (9, 8, 25, 24)
    assert decision.zoom == 32 / 16
    assert decision.fallback is False
    assert decision.clamped is False
    class_average_before = class_average.copy()
    matched_projection_before = matched_projection.copy()
    assert class_average.shape == matched_projection.shape == (32, 32)
    assert np.array_equal(class_average, class_average_before)
    assert np.array_equal(matched_projection, matched_projection_before)


def test_auto_crop_matches_the_foreground_and_surface_maximum_axes():
    matched_projection = np.zeros((32, 32), dtype=np.float32)
    matched_projection[12:20, 13:21] = 1.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [SurfaceSilhouetteBounds(0.20, 0.35, 0.80, 0.65)],
        enabled=True,
    )

    # The raw foreground's maximum axis is 8 px and the silhouette's maximum
    # axis is 60%, so the requested window is ceil(8 / .6) = 14 px.
    assert decision.crop_bounds == (10, 9, 24, 23)
    assert decision.crop_shape == (14, 14)
    assert 8 / 14 == pytest.approx(0.60, abs=0.03)
    assert decision.zoom == 32 / 14


def test_auto_crop_uses_padding_lower_bound_and_records_the_clamp():
    matched_projection = np.zeros((64, 64), dtype=np.float32)
    matched_projection[28:36, 28:36] = 1.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [
            SurfaceSilhouetteBounds(
                left=0.05,
                top=0.05,
                right=0.95,
                bottom=0.95,
            )
        ],
        enabled=True,
    )

    # Raw occupancy cannot reach 90% while retaining the required 2 px
    # padding: the 12 px padded box is the natural lower bound.
    assert decision.crop_bounds == (26, 26, 38, 38)
    assert decision.zoom == 64 / 12
    assert decision.clamped is True
    assert decision.fallback is False


def test_auto_crop_keeps_a_compact_foreground_aligned_with_a_large_silhouette():
    matched_projection = np.zeros((64, 64), dtype=np.float32)
    matched_projection[26:38, 26:38] = 1.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [
            SurfaceSilhouetteBounds(
                left=0.125,
                top=0.125,
                right=0.875,
                bottom=0.875,
            )
        ],
        enabled=True,
    )

    # A 75% target requests 16 px, exactly the padded 16 px box.  The raw
    # foreground therefore occupies the target 75% of the crop.
    assert decision.crop_bounds == (24, 24, 40, 40)
    assert decision.crop_shape == (16, 16)
    assert 12 / 16 == pytest.approx(0.75)
    assert decision.zoom == 64 / 16
    assert decision.clamped is False
    assert decision.fallback is False


def test_auto_crop_matches_raw_foreground_occupancy_when_padding_allows_it():
    matched_projection = np.zeros((128, 128), dtype=np.float32)
    matched_projection[56:72, 56:72] = 1.0
    silhouette = SurfaceSilhouetteBounds(
        left=0.125,
        top=0.125,
        right=0.875,
        bottom=0.875,
    )

    decision = compute_auto_crop_2d_framing(
        [matched_projection], [silhouette], enabled=True
    )

    # Matching the raw 16 px foreground to a 75% silhouette requests a 22 px
    # window.  That window also contains the complete min-2px padding box and
    # requires more than the old fixed 4x cap.
    assert decision.crop_bounds == (53, 53, 75, 75)
    assert decision.crop_shape == (22, 22)
    assert 16 / 22 == pytest.approx(silhouette.width_fraction, abs=0.03)
    assert decision.zoom == 128 / 22
    assert decision.clamped is False
    assert decision.fallback is False


def test_auto_crop_matches_display_visible_occupancy_with_connected_weak_halo():
    size = 128
    rows, columns = np.indices((size, size))
    radius = np.hypot(
        columns - (size - 1) / 2.0,
        rows - (size - 1) / 2.0,
    )
    core_radius = 30.0
    halo_value = 0.38
    matched_projection = np.zeros((size, size), dtype=np.float32)
    core_profile = halo_value + 8.8 * np.maximum(
        0.0,
        1.0 - radius / core_radius,
    ) ** 1.2
    core = radius <= core_radius
    halo = (radius > core_radius) & (radius <= core_radius + 4.0)
    matched_projection[core] = core_profile[core]
    matched_projection[halo] = halo_value
    # A handful of bright peaks are valid signal, not isolated outliers.
    for row, column, value in (
        (62, 63, 10.0),
        (63, 64, 9.7),
        (64, 63, 9.4),
        (61, 64, 9.2),
        (65, 62, 8.9),
        (63, 61, 8.7),
    ):
        matched_projection[row, column] = value

    silhouette = SurfaceSilhouetteBounds(0.125, 0.125, 0.875, 0.875)
    decision = compute_auto_crop_2d_framing(
        [matched_projection], [silhouette], enabled=True
    )

    assert decision.fallback is False
    background = float(
        np.median(
            np.concatenate(
                [
                    matched_projection[0, :],
                    matched_projection[-1, :],
                    matched_projection[1:-1, 0],
                    matched_projection[1:-1, -1],
                ]
            )
        )
    )
    visible = np.abs(matched_projection - background) >= (
        0.05 * np.max(np.abs(matched_projection - background))
    )
    left, top, right, bottom = decision.crop_bounds
    visible_rows, visible_columns = np.nonzero(visible[top:bottom, left:right])
    visible_width = visible_columns.max() - visible_columns.min() + 1
    visible_height = visible_rows.max() - visible_rows.min() + 1
    visible_occupancy = max(visible_width, visible_height) / decision.crop_shape[0]

    assert visible_occupancy == pytest.approx(
        max(silhouette.width_fraction, silhouette.height_fraction),
        abs=0.03,
    )


def test_auto_crop_uses_spatially_supported_peak_to_reject_a_weak_halo():
    size = 128
    matched_projection = np.zeros((size, size), dtype=np.float32)
    matched_projection[56:72, 56:72] = 8.0
    matched_projection[63, 63] = 10.0
    # This halo is connected to the core, but remains below the 5%-of-peak
    # display threshold.  A global percentile can mistake it for foreground.
    matched_projection[40:88, 40:88][matched_projection[40:88, 40:88] == 0] = 0.45
    silhouette = SurfaceSilhouetteBounds(0.125, 0.125, 0.875, 0.875)

    decision = compute_auto_crop_2d_framing(
        [matched_projection], [silhouette], enabled=True
    )

    assert decision.fallback is False
    assert decision.foreground_bounds == (56, 56, 72, 72)
    assert decision.crop_shape == (22, 22)


def test_auto_crop_ignores_a_single_extreme_pixel_in_the_peak_estimate():
    size = 128
    matched_projection = np.zeros((size, size), dtype=np.float32)
    matched_projection[56:72, 56:72] = 8.0
    matched_projection[63, 63] = 10.0
    baseline = compute_auto_crop_2d_framing(
        [matched_projection],
        [SurfaceSilhouetteBounds(0.125, 0.125, 0.875, 0.875)],
        enabled=True,
    )

    # Keep the outlier attached to the object so connected-component-only
    # peak estimates cannot silently treat this as a separate component.
    matched_projection[64, 64] = 1000.0
    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [SurfaceSilhouetteBounds(0.125, 0.125, 0.875, 0.875)],
        enabled=True,
    )

    assert decision.fallback is False
    assert decision.foreground_bounds == baseline.foreground_bounds
    assert decision.crop_bounds == baseline.crop_bounds


def test_auto_crop_falls_back_when_display_foreground_is_compact():
    matched_projection = np.zeros((64, 64), dtype=np.float32)
    # The display-level threshold leaves a 7-pixel-wide core.  The lower seed
    # mask is an invisible halo and must not be used to manufacture a box.
    matched_projection[26:37, 26:34] = 0.1
    matched_projection[28:35, 28:31] = 5.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [SurfaceSilhouetteBounds(0.25, 0.25, 0.75, 0.75)],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "foreground_bbox_too_small"
    assert decision.crop_bounds == (0, 0, 64, 64)


def test_auto_crop_records_non_finite_foreground_fallback_reason():
    matched_projection = np.zeros((32, 32), dtype=np.float32)
    matched_projection[12:20, 13:21] = 1.0
    matched_projection[0, 0] = np.nan

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [SurfaceSilhouetteBounds(0.25, 0.25, 0.75, 0.75)],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "non_finite_projection"


def test_auto_crop_records_invalid_silhouette_fallback_reason():
    matched_projection = np.zeros((32, 32), dtype=np.float32)
    matched_projection[12:20, 13:21] = 1.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [object()],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "invalid_silhouette_bounds"
    assert decision.crop_bounds == (0, 0, 32, 32)


def test_auto_crop_follows_an_off_center_valid_foreground():
    matched_projection = np.zeros((64, 64), dtype=np.float32)
    matched_projection[16:32, 40:56] = 3.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection], [_HALF_FRAME_SILHOUETTE], enabled=True
    )

    assert decision.fallback is False
    assert decision.foreground_bounds == (40, 16, 56, 32)
    assert decision.crop_bounds == (32, 8, 64, 40)
    assert decision.crop_bounds[2] == matched_projection.shape[1]
    assert 16 / 32 == pytest.approx(_HALF_FRAME_SILHOUETTE.width_fraction)


def test_auto_crop_falls_back_when_native_frame_cannot_retain_padding():
    matched_projection = np.zeros((64, 64), dtype=np.float32)
    matched_projection[20:36, 1:17] = 1.0

    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [
            SurfaceSilhouetteBounds(
                left=0.125,
                top=0.125,
                right=0.875,
                bottom=0.875,
            )
        ],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason == "foreground_padding_out_of_bounds"
    assert decision.crop_bounds == (0, 0, 64, 64)


@pytest.mark.parametrize(
    "matched_projection",
    [
        np.zeros((32, 32), dtype=np.float32),
        np.pad(np.ones((1, 1), dtype=np.float32), ((15, 16), (15, 16))),
        np.pad(np.ones((8, 8), dtype=np.float32), ((0, 24), (12, 12))),
        np.pad(np.ones((4, 4), dtype=np.float32), ((14, 14), (14, 14))),
    ],
    ids=["constant", "single-pixel", "boundary", "undersized"],
)
def test_unreliable_foreground_falls_back_to_the_native_frame(matched_projection):
    decision = compute_auto_crop_2d_framing(
        [matched_projection],
        [_HALF_FRAME_SILHOUETTE],
        enabled=True,
    )

    assert decision.fallback is True
    assert decision.fallback_reason is not None
    assert decision.crop_bounds == (0, 0, 32, 32)
