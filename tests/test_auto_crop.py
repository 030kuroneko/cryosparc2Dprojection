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
