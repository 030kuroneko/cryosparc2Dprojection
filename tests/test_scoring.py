import numpy as np
import pytest

from cryosparc_2d_projection.scoring import (
    BandLimitedScoreConfig,
    compute_diagnostic_band_limited_score,
)


def test_diagnostic_band_limited_score_is_one_for_identical_images():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)

    result = compute_diagnostic_band_limited_score(
        image,
        image.copy(),
        pixel_size_A=1.0,
    )

    assert result.valid
    assert np.isclose(result.score, 1.0)
    assert result.metadata["score_role"] == "diagnostic_only"
    assert result.metadata["band_limited_score_valid"]
    assert result.metadata["band_limited_invalid_reason"] is None


def test_diagnostic_band_limited_score_is_invariant_to_positive_affine_intensity():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)

    result = compute_diagnostic_band_limited_score(
        image,
        3.0 * image + 17.0,
        pixel_size_A=1.0,
    )

    assert result.valid
    assert np.isclose(result.score, 1.0)


def test_diagnostic_band_limited_score_suppresses_mismatch_outside_requested_band():
    coordinates = np.arange(64, dtype=float)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    class_average = np.sin(2.0 * np.pi * x / 16.0)
    low_frequency_mismatch = 5.0 * np.sin(2.0 * np.pi * y / 64.0)
    settings = BandLimitedScoreConfig(
        low_resolution_A=20.0,
        high_resolution_A=8.0,
    )

    result = compute_diagnostic_band_limited_score(
        class_average,
        class_average + low_frequency_mismatch,
        pixel_size_A=1.0,
        settings=settings,
    )

    assert result.valid
    assert result.score > 0.98


def test_diagnostic_band_limited_score_suppresses_high_frequency_mismatch():
    coordinates = np.arange(64, dtype=float)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    class_average = np.sin(2.0 * np.pi * x / 16.0)
    high_frequency_mismatch = 5.0 * np.sin(2.0 * np.pi * y / 4.0)
    settings = BandLimitedScoreConfig(
        low_resolution_A=20.0,
        high_resolution_A=8.0,
    )

    result = compute_diagnostic_band_limited_score(
        class_average,
        class_average + high_frequency_mismatch,
        pixel_size_A=1.0,
        settings=settings,
    )

    assert result.valid
    assert result.score > 0.98


def test_diagnostic_band_limited_score_uses_physical_cutoffs_across_pixel_sizes():
    coordinates = np.arange(64, dtype=float)
    settings = BandLimitedScoreConfig(
        low_resolution_A=20.0,
        high_resolution_A=8.0,
    )

    scores = []
    for pixel_size_A, period_pixels in ((1.0, 16), (2.0, 8)):
        x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
        class_average = np.sin(2.0 * np.pi * x / period_pixels)
        mismatch = 5.0 * np.sin(2.0 * np.pi * y / (64 if pixel_size_A == 1 else 32))
        result = compute_diagnostic_band_limited_score(
            class_average,
            class_average + mismatch,
            pixel_size_A=pixel_size_A,
            settings=settings,
        )
        assert result.valid
        assert result.metadata["band_high_resolution_A_effective"] == 8.0
        scores.append(result.score)

    assert np.all(np.asarray(scores) > 0.98)


def test_diagnostic_band_limited_score_clamps_high_resolution_to_nyquist():
    settings = BandLimitedScoreConfig(
        low_resolution_A=20.0,
        high_resolution_A=1.0,
    )
    image = np.sin(
        2.0 * np.pi * np.arange(64, dtype=float)[None, :] / 12.0
    ) + np.zeros((64, 1))

    result = compute_diagnostic_band_limited_score(
        image,
        image.copy(),
        pixel_size_A=2.0,
        settings=settings,
    )

    assert result.valid
    assert result.metadata["band_high_resolution_A_requested"] == 1.0
    assert result.metadata["band_high_resolution_A_effective"] == 4.0
    assert result.metadata["band_high_resolution_A_was_clamped"]


def test_diagnostic_band_limited_score_clamps_low_resolution_to_box_extent():
    coordinates = np.arange(64, dtype=float)
    image = np.sin(2.0 * np.pi * coordinates[None, :] / 32.0) + np.zeros(
        (64, 1)
    )

    result = compute_diagnostic_band_limited_score(
        image,
        image.copy(),
        pixel_size_A=1.0,
    )

    assert result.valid
    assert result.metadata["band_low_resolution_A_requested"] == 80.0
    assert result.metadata["band_low_resolution_A_effective"] == 64.0
    assert result.metadata["band_low_resolution_A_was_clamped"]


def test_diagnostic_band_limited_score_returns_invalid_for_zero_variance():
    image = np.ones((64, 64), dtype=float)

    result = compute_diagnostic_band_limited_score(
        image,
        image.copy(),
        pixel_size_A=1.0,
    )

    assert not result.valid
    assert result.score is None
    assert result.invalid_reason == "zero_weighted_variance"


def test_diagnostic_band_limited_score_returns_invalid_for_empty_passband():
    image = np.arange(8 * 8, dtype=float).reshape(8, 8)

    result = compute_diagnostic_band_limited_score(
        image,
        image.copy(),
        pixel_size_A=1.0,
    )

    assert not result.valid
    assert result.score is None
    assert result.invalid_reason == "no_passband_bins"
    assert result.metadata["band_passband_bin_count"] == 0


def test_diagnostic_band_limited_score_uses_soft_mask_edge_as_weight():
    coordinates = np.arange(64, dtype=float) - 31.5
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    radius = np.hypot(x, y)
    class_average = np.sin(2.0 * np.pi * x / 16.0)
    edge_error = 8.0 * np.sin(2.0 * np.pi * (x + y) / 16.0)
    edge_error *= (radius > 14.0) & (radius < 18.0)

    hard_edge = compute_diagnostic_band_limited_score(
        class_average,
        class_average + edge_error,
        pixel_size_A=1.0,
        settings=BandLimitedScoreConfig(
            low_resolution_A=20.0,
            high_resolution_A=8.0,
            mask_radius_fraction=0.2,
            mask_edge_fraction=0.0,
        ),
    )
    soft_edge = compute_diagnostic_band_limited_score(
        class_average,
        class_average + edge_error,
        pixel_size_A=1.0,
        settings=BandLimitedScoreConfig(
            low_resolution_A=20.0,
            high_resolution_A=8.0,
            mask_radius_fraction=0.2,
            mask_edge_fraction=0.1,
        ),
    )

    assert hard_edge.valid and soft_edge.valid
    assert soft_edge.score < hard_edge.score
    assert soft_edge.metadata["mask_edge_width_px"] == 6.4


@pytest.mark.parametrize(
    "settings",
    [
        {"low_resolution_A": 10.0, "high_resolution_A": 10.0},
        {"low_resolution_A": -80.0},
        {"mask_radius_fraction": 0.0},
        {"mask_edge_fraction": -0.01},
        {"mask_radius_fraction": 0.46, "mask_edge_fraction": 0.05},
        {"filter_order": 0},
        {"filter_type": "brick_wall"},
    ],
)
def test_diagnostic_band_limited_score_config_rejects_unsafe_settings(settings):
    with pytest.raises(ValueError):
        BandLimitedScoreConfig(**settings)
