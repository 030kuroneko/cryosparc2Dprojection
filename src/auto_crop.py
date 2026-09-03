"""Presentation-only automatic framing for native 2D Class Results."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as label_components
from scipy.ndimage import rank_filter

from cryosparc_2d_projection.surface_render import SurfaceSilhouetteBounds


AUTO_CROP_PADDING_FRACTION = 0.10
_MIN_COMPONENT_FRACTION = 0.005
_MIN_BBOX_FRACTION = 0.10
_MIN_BBOX_PIXELS = 5
# Use a high, deterministic percentile so a few extreme pixels do not define
# the display foreground while the threshold still tracks visible signal.
_ROBUST_PEAK_PERCENTILE = 99.9
# Keep a compact valid foreground from being reduced below the accepted bbox
# size when the high percentile has only a few signal samples to work with.
_FALLBACK_ROBUST_PEAK_PERCENTILE = 99.0
# A display peak must have local spatial support.  This removes isolated
# hot-pixels without suppressing the clustered extrema present in projections.
_LOCAL_PEAK_SUPPORT_FRACTION = 0.75
_MIN_LOCAL_PEAK_SUPPORT = 3
# Accepted foreground boxes are at least 10% of the native frame on each
# axis.  With the 10% display padding on both sides, 8 1/3x is the greatest
# zoom that can still retain that accepted padded box.  This is a natural
# safety bound, rather than a presentation-specific fixed zoom choice.
AUTO_CROP_MAX_ZOOM = 1.0 / (
    _MIN_BBOX_FRACTION * (1.0 + 2.0 * AUTO_CROP_PADDING_FRACTION)
)


@dataclass(frozen=True)
class AutoCropDecision:
    """A display crop decision; no scientific image is changed by this object."""

    source_shape: tuple[int, int]
    crop_bounds: tuple[int, int, int, int]
    foreground_bounds: tuple[int, int, int, int] | None
    silhouette_bounds: SurfaceSilhouetteBounds | None
    zoom: float
    clamped: bool
    fallback: bool
    fallback_reason: str | None = None
    enabled: bool = True

    @property
    def crop_shape(self):
        left, top, right, bottom = self.crop_bounds
        return (bottom - top, right - left)

    def as_dict(self):
        left, top, right, bottom = self.crop_bounds
        payload = {
            "enabled": bool(self.enabled),
            "source_shape": [int(value) for value in self.source_shape],
            "crop_bounds": {
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
            },
            "crop_shape": [int(value) for value in self.crop_shape],
            "zoom": float(self.zoom),
            "clamped": bool(self.clamped),
            "fallback": bool(self.fallback),
            "fallback_reason": self.fallback_reason,
        }
        if self.foreground_bounds is None:
            payload["foreground_bounds"] = None
        else:
            foreground_left, foreground_top, foreground_right, foreground_bottom = (
                self.foreground_bounds
            )
            payload["foreground_bounds"] = {
                "left": int(foreground_left),
                "top": int(foreground_top),
                "right": int(foreground_right),
                "bottom": int(foreground_bottom),
            }
        payload["silhouette_bounds"] = (
            None if self.silhouette_bounds is None else self.silhouette_bounds.as_dict()
        )
        return payload


def compute_auto_crop_2d_framing(
    matched_projections,
    silhouette_bounds,
    *,
    enabled,
):
    """Compute one common display crop for one or more matched projections.

    The arrays are only inspected.  The returned bounds are applied as
    Matplotlib display limits; no scientific or display array is sliced.
    """

    projections = _coerce_projections(matched_projections)
    shape = projections[0].shape
    if any(projection.shape != shape for projection in projections[1:]):
        raise ValueError("matched projections must share one square shape")
    height, width = shape
    if type(enabled) is not bool:
        raise ValueError("auto-crop enabled flag must be boolean")
    full_bounds = (0, 0, width, height)
    if not enabled:
        return AutoCropDecision(
            source_shape=shape,
            crop_bounds=full_bounds,
            foreground_bounds=None,
            silhouette_bounds=None,
            zoom=1.0,
            clamped=False,
            fallback=False,
            enabled=False,
        )

    try:
        bounds = _coerce_silhouette_bounds(silhouette_bounds)
    except ValueError:
        return _fallback(shape, "invalid_silhouette_bounds")
    if not bounds:
        return _fallback(shape, "missing_silhouette_bounds")
    try:
        target = _union_silhouette_bounds(bounds)
    except ValueError as error:
        return _fallback(shape, str(error))
    foreground, foreground_failure = _detect_foreground_bounds(projections)
    if foreground is None:
        return _fallback(shape, foreground_failure or "unreliable_foreground")
    foreground_left, foreground_top, foreground_right, foreground_bottom = foreground
    raw_foreground_width = foreground_right - foreground_left
    raw_foreground_height = foreground_bottom - foreground_top
    horizontal_padding = max(
        2.0,
        raw_foreground_width * AUTO_CROP_PADDING_FRACTION,
    )
    vertical_padding = max(
        2.0,
        raw_foreground_height * AUTO_CROP_PADDING_FRACTION,
    )
    required_horizontal_padding = int(np.ceil(horizontal_padding))
    required_vertical_padding = int(np.ceil(vertical_padding))
    if (
        foreground_left < required_horizontal_padding
        or foreground_top < required_vertical_padding
        or width - foreground_right < required_horizontal_padding
        or height - foreground_bottom < required_vertical_padding
    ):
        return _fallback(shape, "foreground_padding_out_of_bounds")
    foreground_width = raw_foreground_width + 2.0 * horizontal_padding
    foreground_height = raw_foreground_height + 2.0 * vertical_padding
    desired_side = max(
        raw_foreground_width,
        raw_foreground_height,
    ) / max(target.width_fraction, target.height_fraction)
    minimum_side = int(
        np.ceil(max(width, height) / AUTO_CROP_MAX_ZOOM)
    )
    requested_side = int(np.ceil(desired_side))
    padding_side = int(np.ceil(max(foreground_width, foreground_height)))
    minimum_side = max(minimum_side, padding_side)
    clamped = requested_side < minimum_side
    side = min(max(width, height), max(minimum_side, requested_side))
    if side >= width or side >= height:
        return AutoCropDecision(
            source_shape=shape,
            crop_bounds=full_bounds,
            foreground_bounds=foreground,
            silhouette_bounds=target,
            zoom=1.0,
            clamped=False,
            fallback=False,
            enabled=True,
        )
    center_x = (foreground_left + foreground_right) / 2.0
    center_y = (foreground_top + foreground_bottom) / 2.0
    left = _centered_start(center_x, side, width)
    top = _centered_start(center_y, side, height)
    crop_bounds = (left, top, left + side, top + side)
    return AutoCropDecision(
        source_shape=shape,
        crop_bounds=crop_bounds,
        foreground_bounds=foreground,
        silhouette_bounds=target,
        zoom=float(width / side),
        clamped=clamped,
        fallback=False,
        enabled=True,
    )


def set_auto_crop_2d_limits(axis, decision):
    """Apply a display-only crop by setting Matplotlib image limits.

    ``decision.crop_bounds`` uses half-open array coordinates.  Matplotlib's
    pixel centers are offset by half a pixel, and image rows use an inverted
    y-axis, so the limits preserve the exact selected pixel window while the
    underlying array remains native and unchanged.
    """
    if not isinstance(decision, AutoCropDecision):
        raise TypeError("decision must be an AutoCropDecision")
    left, top, right, bottom = decision.crop_bounds
    try:
        axis.set_xlim(left - 0.5, right - 0.5)
        axis.set_ylim(bottom - 0.5, top - 0.5)
    except AttributeError as error:
        raise TypeError("axis must expose set_xlim and set_ylim") from error


def _coerce_projections(matched_projections):
    if isinstance(matched_projections, np.ndarray) and matched_projections.ndim == 2:
        projections = (np.asarray(matched_projections, dtype=np.float32),)
    else:
        try:
            projections = tuple(
                np.asarray(projection, dtype=np.float32)
                for projection in matched_projections
            )
        except TypeError as error:
            raise ValueError("matched projections must be a non-empty sequence") from error
    if not projections or any(
        projection.ndim != 2
        or projection.shape[0] != projection.shape[1]
        or projection.shape[0] < 2
        for projection in projections
    ):
        raise ValueError("matched projections must be non-empty square 2D images")
    return projections


def _coerce_silhouette_bounds(values):
    if isinstance(values, SurfaceSilhouetteBounds):
        return (values,)
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("silhouette bounds must be a bound or sequence") from error
    if not values:
        return ()
    if not all(isinstance(value, SurfaceSilhouetteBounds) for value in values):
        raise ValueError("silhouette bounds must use SurfaceSilhouetteBounds")
    return values


def _union_silhouette_bounds(values):
    return SurfaceSilhouetteBounds(
        left=min(value.left for value in values),
        top=min(value.top for value in values),
        right=max(value.right for value in values),
        bottom=max(value.bottom for value in values),
    )


def _detect_foreground_bounds(projections):
    detected = []
    for projection in projections:
        if not np.isfinite(projection).all():
            return None, "non_finite_projection"
        border = np.concatenate(
            [
                projection[0, :],
                projection[-1, :],
                projection[1:-1, 0],
                projection[1:-1, -1],
            ]
        )
        background = float(np.median(border))
        deviations = np.abs(projection - background)
        mad = float(np.median(np.abs(border - background)))
        bounds, failure = _foreground_bounds_for_peak(
            deviations,
            projection.shape,
            _ROBUST_PEAK_PERCENTILE,
            mad,
        )
        if bounds is None and failure == "foreground_bbox_too_small":
            bounds, failure = _foreground_bounds_for_peak(
                deviations,
                projection.shape,
                _FALLBACK_ROBUST_PEAK_PERCENTILE,
                mad,
            )
        if bounds is None:
            return None, failure
        detected.append(bounds)
    return (
        (
            min(bounds[0] for bounds in detected),
            min(bounds[1] for bounds in detected),
            max(bounds[2] for bounds in detected),
            max(bounds[3] for bounds in detected),
        ),
        None,
    )


def _foreground_bounds_for_peak(deviations, shape, percentile, mad):
    provisional_peak = float(np.percentile(deviations, percentile))
    threshold = max(6.0 * mad, 0.05 * provisional_peak)
    if threshold <= 0.0 or not np.isfinite(threshold):
        return None, "constant_or_low_dynamic_range"
    largest, failure = _largest_component(deviations >= threshold)
    if largest is None:
        return None, failure

    # Estimate the display peak from a spatially supported part of the
    # dominant signal component.  The provisional mask supplies the component
    # identity; the local support check prevents one extreme pixel, including
    # one attached to the component, from setting the final threshold.
    supported_peak = _spatially_supported_peak(deviations, largest)
    if supported_peak is not None:
        threshold = max(6.0 * mad, 0.05 * supported_peak)
    foreground = deviations >= threshold
    largest, failure = _largest_component(foreground)
    if largest is None:
        return None, failure

    bounds, failure = _bounds_for_component(largest, shape)
    return (bounds, None) if bounds is not None else (None, failure)


def _bounds_for_component(component, shape):
    area = int(np.count_nonzero(component))
    minimum_component_area = max(
        1,
        int(np.ceil(np.prod(shape) * _MIN_COMPONENT_FRACTION)),
    )
    if area < minimum_component_area:
        return None, "foreground_component_too_small"
    rows, columns = np.nonzero(component)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    height, width = shape
    if left == 0 or top == 0 or right == width or bottom == height:
        return None, "foreground_touches_boundary"
    minimum_bbox_width = max(_MIN_BBOX_PIXELS, int(np.ceil(width * _MIN_BBOX_FRACTION)))
    minimum_bbox_height = max(
        _MIN_BBOX_PIXELS,
        int(np.ceil(height * _MIN_BBOX_FRACTION)),
    )
    if right - left < minimum_bbox_width or bottom - top < minimum_bbox_height:
        return None, "foreground_bbox_too_small"
    return (left, top, right, bottom), None


def _largest_component(mask):
    components, count = label_components(mask)
    if count == 0:
        return None, "no_foreground_component"
    sizes = np.bincount(components.ravel())[1:]
    largest_label = int(np.argmax(sizes)) + 1
    return components == largest_label, None


def _spatially_supported_peak(deviations, component):
    local_rank = rank_filter(
        deviations,
        rank=9 - _MIN_LOCAL_PEAK_SUPPORT,
        size=3,
        mode="constant",
        cval=0.0,
    )
    supported = component & (deviations > 0.0) & (
        local_rank >= _LOCAL_PEAK_SUPPORT_FRACTION * deviations
    )
    if not np.any(supported):
        return None
    return float(np.max(deviations[supported]))


def _centered_start(center, side, extent):
    start = int(np.rint(center - side / 2.0))
    return min(max(0, start), extent - side)


def _fallback(shape, reason):
    height, width = shape
    return AutoCropDecision(
        source_shape=shape,
        crop_bounds=(0, 0, width, height),
        foreground_bounds=None,
        silhouette_bounds=None,
        zoom=1.0,
        clamped=False,
        fallback=True,
        fallback_reason=str(reason),
        enabled=True,
    )
