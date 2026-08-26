"""Static five-column presentation for Symmetry-Axis Search results."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate as rotate_image

from cryosparc_2d_projection.axis_registry import get_axis_family


AXIS_RESULT_COLUMNS = (
    "Axis-Aligned Class",
    "Best Near-Axis Projection",
    "Exact Axis Projection",
    "Best Near-Axis 3D View",
    "Exact Axis 3D View",
)


@dataclass(frozen=True)
class AxisResultPanelRow:
    family_name: str
    rank: int
    class_number: int
    axis_aligned_class: np.ndarray
    near_axis_projection: np.ndarray
    exact_axis_projection: np.ndarray
    near_axis_view: np.ndarray
    exact_axis_view: np.ndarray

    def panels(self):
        return (
            self.axis_aligned_class,
            self.near_axis_projection,
            self.exact_axis_projection,
            self.near_axis_view,
            self.exact_axis_view,
        )


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
):
    """Create the approved static five-column result figure."""

    from matplotlib.figure import Figure

    rows = tuple(rows)
    if not rows:
        raise ValueError("at least one Axis Result row is required")
    axis_rolls = dict(axis_rolls or {})
    figure = Figure(
        figsize=(15, 3 * len(rows)),
        dpi=int(dpi),
        constrained_layout=True,
    )
    for row_index, row in enumerate(rows):
        roll = float(axis_rolls.get(row.family_name, 0.0))
        for column_index, (title, panel) in enumerate(
            zip(AXIS_RESULT_COLUMNS, row.panels(), strict=True)
        ):
            axis = figure.add_subplot(
                len(rows),
                len(AXIS_RESULT_COLUMNS),
                row_index * len(AXIS_RESULT_COLUMNS) + column_index + 1,
            )
            displayed = apply_axis_display_roll(
                panel,
                roll,
                background=background,
            )
            axis.imshow(displayed, cmap="gray" if displayed.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")
    return figure
