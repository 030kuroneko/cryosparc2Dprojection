import numpy as np
from matplotlib.text import Text

from cryosparc_2d_projection.axis_presentation import (
    AXIS_RESULT_COLUMNS,
    AxisResultLabel,
    AxisResultPanelRow,
    ExactAxisResultPanelRow,
    apply_axis_display_roll,
    create_axis_result_figure,
    parse_axis_rolls,
)


def _marker_panel():
    panel = np.zeros((9, 9), dtype=np.float32)
    panel[4, 7] = 1.0
    return panel


def test_positive_axis_roll_is_visually_counterclockwise_and_preserves_box():
    rotated = apply_axis_display_roll(_marker_panel(), 90.0, background="dark")

    assert rotated.shape == (9, 9)
    assert np.unravel_index(np.argmax(rotated), rotated.shape) == (1, 4)


def test_axis_result_figure_has_exactly_five_columns_in_approved_order():
    panel = _marker_panel()
    row = AxisResultPanelRow(
        label=AxisResultLabel(
            family_name="2fold",
            rank=1,
            class_number=3,
            axis_class_score=0.0,
        ),
        axis_aligned_class=panel,
        near_axis_projection=panel,
        exact_axis_projection=panel,
        near_axis_view=panel,
        exact_axis_view=panel,
    )

    figure = create_axis_result_figure(
        [row],
        axis_rolls={"2fold": 10.0},
        dpi=100,
        background="dark",
    )

    assert len(figure.axes) == 5
    assert tuple(axis.get_title() for axis in figure.axes) == AXIS_RESULT_COLUMNS
    assert all(axis.images[0].get_array().shape[:2] == (9, 9) for axis in figure.axes)


def test_axis_result_row_label_reports_family_rank_class_and_scores_outside_panels():
    panel = _marker_panel()
    row = AxisResultPanelRow(
        label=AxisResultLabel(
            family_name="5fold",
            rank=2,
            class_number=21,
            axis_class_score=0.87321,
            near_axis_score=0.88104,
            near_axis_angle_degrees=3.257,
        ),
        axis_aligned_class=panel,
        near_axis_projection=panel,
        exact_axis_projection=panel,
        near_axis_view=panel,
        exact_axis_view=panel,
    )

    figure = create_axis_result_figure([row])

    label = next(
        text
        for text in figure.findobj(Text)
        if text.get_text()
        == (
            "5fold · Rank 2 · Class 21 · Score 0.8732 "
            "· Near 0.8810 · Near-Axis Angle 3.257°"
        )
    )
    assert label.axes is None
    assert all(label not in axis.texts for axis in figure.axes)


def test_exact_axis_result_row_label_omits_near_score():
    panel = _marker_panel()
    row = ExactAxisResultPanelRow(
        label=AxisResultLabel(
            family_name="3fold",
            rank=1,
            class_number=8,
            axis_class_score=0.76543,
        ),
        class_average=panel,
        exact_matched_projection=panel,
        exact_axis_view=panel,
    )

    figure = create_axis_result_figure([row])

    labels = {text.get_text() for text in figure.findobj(Text)}
    assert "3fold · Rank 1 · Class 8 · Score 0.7654" in labels
    assert not any("Near" in label for label in labels)


def test_repeatable_axis_roll_values_are_parsed_per_family():
    assert parse_axis_rolls(["2fold=10", "3fold=-5"]) == {
        "2fold": 10.0,
        "3fold": -5.0,
    }
