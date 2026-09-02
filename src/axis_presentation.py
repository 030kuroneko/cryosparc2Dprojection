"""Static five-column presentation for Symmetry-Axis Search results."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate as rotate_image

from cryosparc_2d_projection.auto_crop import (
    AutoCropDecision,
    set_auto_crop_2d_limits,
)
from cryosparc_2d_projection.axis_registry import get_axis_family
from cryosparc_2d_projection.presentation import ComparisonRenderOptions


AXIS_RESULT_COLUMNS = (
    "Class Average",
    "Near-Axis Matched Projection",
    "Exact-Axis Matched Projection",
    "Near-Axis Camera View",
    "Exact-Axis Camera View",
)

EXACT_AXIS_RESULT_COLUMNS = (
    "Class Average",
    "Exact-Axis Matched Projection",
    "Exact-Axis Camera View",
)


@dataclass(frozen=True)
class AxisResultLabel:
    family_name: str
    rank: int
    class_number: int
    axis_class_score: float
    near_axis_score: float | None = None
    near_axis_angle_degrees: float | None = None

    def text(self):
        result = (
            f"{self.family_name} · Rank {self.rank} · Class {self.class_number} "
            f"· Score {self.axis_class_score:.4f}"
        )
        if self.near_axis_score is not None:
            result += f" · Near {self.near_axis_score:.4f}"
        if self.near_axis_angle_degrees is not None:
            result += (
                f" · Near-Axis Angle {self.near_axis_angle_degrees:.3f}°"
            )
        return result


@dataclass(frozen=True)
class ExactAxisResultPanelRow:
    label: AxisResultLabel
    class_average: np.ndarray
    exact_matched_projection: np.ndarray
    exact_axis_view: np.ndarray
    auto_crop_decision: AutoCropDecision | None = None

    def panels(self):
        return (
            self.class_average,
            self.exact_matched_projection,
            self.exact_axis_view,
        )

    @property
    def columns(self):
        return EXACT_AXIS_RESULT_COLUMNS


@dataclass(frozen=True)
class AxisResultPanelRow:
    label: AxisResultLabel
    axis_aligned_class: np.ndarray
    near_axis_projection: np.ndarray
    exact_axis_projection: np.ndarray
    near_axis_view: np.ndarray
    exact_axis_view: np.ndarray
    auto_crop_decision: AutoCropDecision | None = None

    def panels(self):
        return (
            self.axis_aligned_class,
            self.near_axis_projection,
            self.exact_axis_projection,
            self.near_axis_view,
            self.exact_axis_view,
        )

    @property
    def columns(self):
        return AXIS_RESULT_COLUMNS


def parse_axis_rolls(values):
    """Parse repeatable ``family=degrees`` display-only overrides."""

    result = {}
    for value in values or ():
        try:
            family_name, degrees_text = value.split("=", 1)
            family = get_axis_family("I", family_name.strip()).name
            degrees = float(degrees_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "axis roll must use FAMILY=DEGREES with 2fold, 3fold, or 5fold"
            ) from error
        if not np.isfinite(degrees):
            raise ValueError("axis roll degrees must be finite")
        if family in result:
            raise ValueError(f"axis roll for {family} was provided more than once")
        result[family] = degrees
    return result


def apply_axis_display_roll(image, degrees, *, background):
    """Rotate one final display panel counterclockwise without resizing it."""

    image = np.asarray(image)
    if image.ndim not in {2, 3} or image.shape[0] != image.shape[1]:
        raise ValueError("axis result panels must be square 2D or RGB images")
    if background not in {"dark", "light"}:
        raise ValueError("axis result background must be dark or light")
    if not np.isfinite(degrees):
        raise ValueError("axis roll degrees must be finite")
    if background == "dark":
        fill = 0.0
    elif np.issubdtype(image.dtype, np.integer):
        fill = float(np.iinfo(image.dtype).max)
    else:
        fill = 1.0
    return rotate_image(
        image,
        float(degrees),
        axes=(1, 0),
        reshape=False,
        order=1,
        mode="constant",
        cval=fill,
        prefilter=False,
    )


def create_axis_result_figure(
    rows,
    *,
    axis_rolls=None,
    dpi=100,
    background="dark",
    comparison_options=None,
):
    """Create the approved static five-column result figure."""

    from matplotlib.figure import Figure

    rows = tuple(rows)
    if not rows:
        raise ValueError("at least one Axis Result row is required")
    axis_rolls = dict(axis_rolls or {})
    comparison_options = comparison_options or ComparisonRenderOptions()
    columns = rows[0].columns
    if any(row.columns != columns for row in rows):
        raise ValueError("all Axis Result rows must use the same columns")
    figure = Figure(
        figsize=(3 * len(columns), 3 * len(rows)),
        dpi=int(dpi),
        constrained_layout=True,
    )
    subfigures = figure.subfigures(
        nrows=len(rows), ncols=1, squeeze=False
    )
    for row_index, row in enumerate(rows):
        roll = float(axis_rolls.get(row.label.family_name, 0.0))
        score_text = row.label.text()
        subfigure = subfigures[row_index, 0]
        subfigure.suptitle(score_text)
        panel_axes = np.asarray(
            subfigure.subplots(nrows=1, ncols=len(columns), squeeze=False)
        )[0]
        for column_index, (title, panel) in enumerate(
            zip(columns, row.panels(), strict=True)
        ):
            axis = panel_axes[column_index]
            displayed = apply_axis_display_roll(
                panel,
                roll,
                background=background,
            )
            decision = getattr(row, "auto_crop_decision", None)
            crop_panel = (
                comparison_options.auto_crop_2d
                and decision is not None
                and column_index < (2 if columns == EXACT_AXIS_RESULT_COLUMNS else 3)
            )
            axis.imshow(displayed, cmap="gray" if displayed.ndim == 2 else None)
            if crop_panel:
                set_auto_crop_2d_limits(axis, decision)
            axis.set_title(title)
            axis.axis("off")
    return figure
