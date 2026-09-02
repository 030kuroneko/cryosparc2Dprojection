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

    # The 8 px foreground receives the minimum 2 px padding on every side;
    # matching a 50% silhouette therefore needs a 24 px display window.
    assert decision.crop_bounds == (5, 4, 29, 28)
    assert decision.zoom == 32 / 24
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
        [SurfaceSilhouetteBounds(0.10, 0.30, 0.90, 0.70)],
        enabled=True,
    )

    # The padded foreground is 12 px on its maximum axis and the silhouette
    # occupies 80% on its maximum axis, so the display window is 15 px.
    assert decision.crop_shape == (15, 15)
    assert decision.zoom == 32 / 15


def test_auto_crop_clamps_zoom_to_three_times_and_records_the_clamp():
    matched_projection = np.zeros((64, 64), dtype=np.float32)
    matched_projection[26:38, 26:38] = 1.0

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

    assert decision.crop_bounds == (21, 21, 43, 43)
    assert decision.zoom == 64 / 22
    assert decision.clamped is True
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
    assert decision.crop_bounds == (24, 4, 64, 44)
    assert decision.crop_bounds[2] == matched_projection.shape[1]


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
