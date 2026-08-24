from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BandLimitedScoreConfig:
    """Configuration for the diagnostic class/projection similarity score."""

    low_resolution_A: float = 80.0
    high_resolution_A: float = 15.0
    mask_radius_fraction: float = 0.45
    mask_edge_fraction: float = 0.05
    filter_order: int = 4
    filter_type: str = "butterworth"

    def __post_init__(self):
        if (
            not np.isfinite(self.low_resolution_A)
            or not np.isfinite(self.high_resolution_A)
            or self.low_resolution_A <= self.high_resolution_A
            or self.high_resolution_A <= 0
        ):
            raise ValueError(
                "low resolution must be finite, positive, and greater than high resolution"
            )
        if (
            not np.isfinite(self.mask_radius_fraction)
            or self.mask_radius_fraction <= 0
        ):
            raise ValueError("mask radius fraction must be finite and positive")
        if (
            not np.isfinite(self.mask_edge_fraction)
            or self.mask_edge_fraction < 0
        ):
            raise ValueError("mask edge fraction must be finite and non-negative")
        if self.mask_radius_fraction + self.mask_edge_fraction > 0.5:
            raise ValueError("mask radius plus edge must not exceed half the box width")
        if not isinstance(self.filter_order, int) or self.filter_order <= 0:
            raise ValueError("filter order must be a positive integer")
        if self.filter_type != "butterworth":
            raise ValueError("v1 diagnostic scoring only supports butterworth filtering")


@dataclass(frozen=True)
class DiagnosticBandLimitedScore:
    """Diagnostic score and the metadata needed to audit its calculation."""

    score: float | None
    valid: bool
    invalid_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def compute_diagnostic_band_limited_score(
    class_average,
    matched_projection,
    *,
    pixel_size_A,
    settings=None,
):
    """Calculate a diagnostic similarity score for a class/projection pair.

    The public seam deliberately returns metadata with the score.  The score is
    diagnostic evidence for the already selected camera and does not rank
    cameras.
    """
    class_average = np.asarray(class_average, dtype=float)
    matched_projection = np.asarray(matched_projection, dtype=float)
    if settings is None:
        settings = BandLimitedScoreConfig()
    if not isinstance(settings, BandLimitedScoreConfig):
        raise TypeError("settings must be a BandLimitedScoreConfig")
    if class_average.ndim != 2 or class_average.shape[0] != class_average.shape[1]:
        raise ValueError("class average must be a square 2D image")
    if matched_projection.shape != class_average.shape:
        raise ValueError(
            "class average and matched projection must have the same shape"
        )
    pixel_size_A = float(pixel_size_A)
    if pixel_size_A <= 0:
        raise ValueError("pixel size must be positive")

    size = class_average.shape[0]
    nyquist_resolution_A = 2.0 * pixel_size_A
    effective_low_resolution_A = min(
        float(settings.low_resolution_A), size * pixel_size_A
    )
    effective_high_resolution_A = max(
        float(settings.high_resolution_A), nyquist_resolution_A
    )
    frequencies = np.fft.fftfreq(size, d=pixel_size_A)
    frequency_y, frequency_x = np.meshgrid(frequencies, frequencies, indexing="ij")
    radial_frequency = np.hypot(frequency_x, frequency_y)
    low_frequency = 1.0 / effective_low_resolution_A
    high_frequency = 1.0 / effective_high_resolution_A
    passband = (
        (radial_frequency >= low_frequency)
        & (radial_frequency <= high_frequency)
        & (radial_frequency > 0)
    )
    passband_bin_count = int(np.count_nonzero(passband))
    mask = _soft_circular_weights(
        size,
        inner_radius=size * settings.mask_radius_fraction,
        edge_width=size * settings.mask_edge_fraction,
    )
    metadata = _base_metadata(
        size,
        pixel_size_A,
        settings,
        mask,
        effective_low_resolution_A=effective_low_resolution_A,
        effective_high_resolution_A=effective_high_resolution_A,
        passband_bin_count=passband_bin_count,
    )
    if passband_bin_count == 0:
        return _make_result(
            score=None,
            valid=False,
            invalid_reason="no_passband_bins",
            metadata=metadata,
        )
    if not np.isfinite(class_average).all() or not np.isfinite(
        matched_projection
    ).all():
        return _make_result(
            score=None,
            valid=False,
            invalid_reason="non_finite_input",
            metadata=metadata,
        )
    frequency_response = _butterworth_bandpass(
        radial_frequency,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        order=settings.filter_order,
    )
    filtered_class_average = _apply_frequency_response(
        class_average, frequency_response
    )
    filtered_projection = _apply_frequency_response(
        matched_projection, frequency_response
    )
    score = _weighted_zero_mean_ncc(
        filtered_class_average,
        filtered_projection,
        mask,
    )
    if score is None:
        return _make_result(
            score=None,
            valid=False,
            invalid_reason="zero_weighted_variance",
            metadata=metadata,
        )
    return _make_result(
        score=score,
        valid=True,
        metadata=metadata,
    )


def _soft_circular_weights(size, *, inner_radius, edge_width):
    coordinates = np.arange(size, dtype=float) - (size - 1) / 2.0
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.hypot(x, y)
    weights = np.zeros_like(radius)
    weights[radius <= inner_radius] = 1.0
    if edge_width > 0:
        transition = (radius > inner_radius) & (radius < inner_radius + edge_width)
        fraction = (radius[transition] - inner_radius) / edge_width
        weights[transition] = 0.5 * (1.0 + np.cos(np.pi * fraction))
    return weights


def _weighted_zero_mean_ncc(left, right, weights):
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return None
    left_mean = float(np.sum(weights * left) / weight_sum)
    right_mean = float(np.sum(weights * right) / weight_sum)
    left_centered = left - left_mean
    right_centered = right - right_mean
    left_variance = float(np.sum(weights * left_centered**2))
    right_variance = float(np.sum(weights * right_centered**2))
    denominator = np.sqrt(left_variance * right_variance)
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    numerator = float(np.sum(weights * left_centered * right_centered))
    if not np.isfinite(numerator):
        return None
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def _butterworth_bandpass(radial_frequency, *, low_frequency, high_frequency, order):
    if order <= 0:
        raise ValueError("filter order must be positive")
    safe_frequency = np.maximum(radial_frequency, np.finfo(float).eps)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        high_pass = 1.0 / (1.0 + (low_frequency / safe_frequency) ** (2 * order))
        low_pass = 1.0 / (1.0 + (safe_frequency / high_frequency) ** (2 * order))
    return high_pass * low_pass


def _apply_frequency_response(image, frequency_response):
    transformed = np.fft.fft2(image)
    return np.fft.ifft2(transformed * frequency_response).real


def _base_metadata(
    size,
    pixel_size_A,
    settings,
    mask,
    *,
    effective_low_resolution_A,
    effective_high_resolution_A,
    passband_bin_count,
):
    return {
        "score_role": "diagnostic_only",
        "score_definition": "weighted_zero_mean_ncc_after_fourier_bandpass",
        "candidate_set_scope": "raw_search_winner",
        "band_low_resolution_A_requested": float(settings.low_resolution_A),
        "band_high_resolution_A_requested": float(settings.high_resolution_A),
        "band_low_resolution_A_effective": float(effective_low_resolution_A),
        "band_low_resolution_A_was_clamped": bool(
            float(settings.low_resolution_A)
            > float(effective_low_resolution_A)
        ),
        "band_high_resolution_A_effective": float(effective_high_resolution_A),
        "band_high_resolution_A_was_clamped": bool(
            float(settings.high_resolution_A)
            < float(effective_high_resolution_A)
        ),
        "band_filter_type": settings.filter_type,
        "band_filter_order": int(settings.filter_order),
        "band_passband_bin_count": int(passband_bin_count),
        "matching_pixel_size_A": float(pixel_size_A),
        "matching_box_size": int(size),
        "matching_box_size_A": float(size * pixel_size_A),
        "matching_nyquist_resolution_A": float(2.0 * pixel_size_A),
        "mask_shape": "soft_circle",
        "mask_radius_px": float(size * settings.mask_radius_fraction),
        "mask_edge_width_px": float(size * settings.mask_edge_fraction),
        "mask_weight_sum": float(np.sum(mask)),
        "mask_effective_pixel_count": float(np.sum(mask**2)),
    }


def _make_result(*, score, valid, metadata, invalid_reason=None):
    enriched_metadata = dict(metadata)
    enriched_metadata["band_limited_score_valid"] = bool(valid)
    enriched_metadata["band_limited_invalid_reason"] = invalid_reason
    return DiagnosticBandLimitedScore(
        score=score,
        valid=valid,
        invalid_reason=invalid_reason,
        metadata=enriched_metadata,
    )
