"""Presentation-only automatic framing for native 2D Class Results."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalCameraView:
    """Physical framing information shared by one or more 2D panels.

    ``camera_viewport_A`` is the square orthographic viewport of the matching
    Camera View Render.  Projection shifts use raw native-array coordinates:
    ``(x, y)`` with rows increasing downward.  The final display flips rows,
    and an optional display roll is applied after that flip.
    """

    camera_viewport_A: float
    projection_shift_pixels: tuple[float, float] = (0.0, 0.0)
    display_roll_degrees: float = 0.0

    def __post_init__(self):
        try:
            viewport = float(self.camera_viewport_A)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "camera viewport must be a finite positive value"
            ) from error
        if not np.isfinite(viewport) or viewport <= 0.0:
            raise ValueError("camera viewport must be a finite positive value")

        try:
            shift = tuple(float(value) for value in self.projection_shift_pixels)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "projection shift must contain two finite values"
            ) from error
        if len(shift) != 2 or not all(np.isfinite(value) for value in shift):
            raise ValueError("projection shift must contain two finite values")

        try:
            roll = float(self.display_roll_degrees)
        except (TypeError, ValueError) as error:
            raise ValueError("display roll must be finite") from error
        if not np.isfinite(roll):
            raise ValueError("display roll must be finite")

        object.__setattr__(self, "camera_viewport_A", viewport)
        object.__setattr__(self, "projection_shift_pixels", shift)
        object.__setattr__(self, "display_roll_degrees", roll)

    def as_dict(self):
        return {
            "camera_viewport_A": float(self.camera_viewport_A),
            "projection_shift_pixels": [
                float(value) for value in self.projection_shift_pixels
            ],
            "display_roll_degrees": float(self.display_roll_degrees),
        }


@dataclass(frozen=True)
class AutoCropDecision:
    """A display crop decision; no scientific image is changed by this object."""

    source_shape: tuple[int, int]
    crop_bounds: tuple[int, int, int, int]
    foreground_bounds: tuple[int, int, int, int] | None
    silhouette_bounds: object | None
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
            None
            if self.silhouette_bounds is None
            else self.silhouette_bounds.as_dict()
        )
        return payload


def compute_auto_crop_2d_framing(
    source_shape,
    native_pixel_size_A,
    view_framings,
    *,
    enabled,
):
    """Compute a display crop from the Camera View Render's physical FOV.

    The crop side is the camera's square physical viewport converted to native
    pixels.  All views use that common side; when multiple views are present,
    their display centers are merged with the midpoint of the per-axis center
    range.  Input image arrays are intentionally absent from this seam, so
    contrast, noise, and display interpolation cannot affect framing.
    """

    shape = _coerce_source_shape(source_shape)
    full_bounds = (0, 0, shape[1], shape[0])
    if type(enabled) is not bool:
        raise ValueError("auto-crop enabled flag must be boolean")
    try:
        pixel_size = _coerce_native_pixel_size(native_pixel_size_A)
    except (OverflowError, TypeError, ValueError):
        if enabled:
            return _fallback(shape, "invalid_native_pixel_size")
        raise
    if not enabled:
        return _decision(
            shape,
            full_bounds,
            zoom=1.0,
            fallback=False,
            enabled=False,
        )

    try:
        views = _coerce_view_framings(view_framings)
    except (TypeError, ValueError):
        return _fallback(shape, "invalid_view_framing")
    if not views:
        return _fallback(shape, "missing_view_framing")

    try:
        requested_side = max(view.camera_viewport_A for view in views) / pixel_size
        if not np.isfinite(requested_side):
            return _fallback(shape, "invalid_camera_viewport")
        side = int(np.ceil(requested_side))
    except (OverflowError, ValueError):
        return _fallback(shape, "invalid_camera_viewport")
    if side < 2:
        return _fallback(shape, "camera_viewport_too_small")
    if side > shape[0] or side > shape[1]:
        return _fallback(shape, "camera_viewport_out_of_bounds")

    centers = np.asarray(
        [_display_center(shape, view) for view in views],
        dtype=float,
    )
    if not np.isfinite(centers).all():
        return _fallback(shape, "invalid_view_center")
    if (
        centers[:, 0].min() < 0.0
        or centers[:, 0].max() > shape[1] - 1.0
        or centers[:, 1].min() < 0.0
        or centers[:, 1].max() > shape[0] - 1.0
    ):
        return _fallback(shape, "view_center_out_of_bounds")
    if (
        centers[:, 0].max() - centers[:, 0].min() > side
        or centers[:, 1].max() - centers[:, 1].min() > side
    ):
        return _fallback(shape, "view_center_range_exceeds_crop")
    shared_center = np.array(
        [
            (centers[:, 0].min() + centers[:, 0].max()) / 2.0,
            (centers[:, 1].min() + centers[:, 1].max()) / 2.0,
        ],
        dtype=float,
    )
    # Bounds are half-open pixel indices.  Their selected pixel centers span
    # ``left`` through ``left + side - 1``, so center the span using its
    # half-width rather than the geometric half-side.  np.rint is deliberate:
    # it gives deterministic nearest-integer, ties-to-even rounding when a
    # half-pixel center cannot be represented by integer bounds.
    half_pixel_span = (side - 1) / 2.0
    left = int(np.rint(shared_center[0] - half_pixel_span))
    top = int(np.rint(shared_center[1] - half_pixel_span))
    right = left + side
    bottom = top + side
    if left < 0 or top < 0 or right > shape[1] or bottom > shape[0]:
        return _fallback(shape, "crop_out_of_bounds")

    return _decision(
        shape,
        (left, top, right, bottom),
        zoom=float(shape[1] / side),
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


def _coerce_source_shape(source_shape):
    try:
        shape = tuple(source_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("source shape must contain two dimensions") from error
    if (
        len(shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 2
            for value in shape
        )
        or int(shape[0]) != int(shape[1])
    ):
        raise ValueError("source shape must be a square 2D shape of at least 2")
    return (int(shape[0]), int(shape[1]))


def _coerce_native_pixel_size(native_pixel_size_A):
    try:
        pixel_size = float(native_pixel_size_A)
    except (TypeError, ValueError) as error:
        raise ValueError("native pixel size must be a finite positive value") from error
    if not np.isfinite(pixel_size) or pixel_size <= 0.0:
        raise ValueError("native pixel size must be a finite positive value")
    return pixel_size


def _coerce_view_framings(values):
    if isinstance(values, PhysicalCameraView):
        return (values,)
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("view framings must be a view or sequence") from error
    if not all(isinstance(value, PhysicalCameraView) for value in values):
        raise ValueError("view framings must use PhysicalCameraView")
    return values


def _display_center(shape, view):
    height, width = shape
    shift_x, shift_y_raw = view.projection_shift_pixels
    # Native projection rows increase downward; the displayed image is
    # vertically flipped before any optional counter-clockwise display roll.
    shift_x_display = float(shift_x)
    shift_y_display = -float(shift_y_raw)
    radians = np.deg2rad(view.display_roll_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    rolled_x = cosine * shift_x_display + sine * shift_y_display
    rolled_y = -sine * shift_x_display + cosine * shift_y_display
    return (
        (width - 1.0) / 2.0 + rolled_x,
        (height - 1.0) / 2.0 + rolled_y,
    )


def _decision(shape, crop_bounds, *, zoom, fallback, enabled, reason=None):
    return AutoCropDecision(
        source_shape=shape,
        crop_bounds=crop_bounds,
        foreground_bounds=None,
        silhouette_bounds=None,
        zoom=float(zoom),
        clamped=False,
        fallback=bool(fallback),
        fallback_reason=reason,
        enabled=bool(enabled),
    )


def _fallback(shape, reason):
    height, width = shape
    return _decision(
        shape,
        (0, 0, width, height),
        zoom=1.0,
        fallback=True,
        enabled=True,
        reason=str(reason),
    )
