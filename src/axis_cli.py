"""Command-line entry point for image-only Symmetry-Axis Class Search."""

import argparse

from cryosparc_2d_projection.axis_external_job import (
    AxisSourceOutput,
    run_axis_search_job,
)
from cryosparc_2d_projection.axis_presentation import parse_axis_rolls
from cryosparc_2d_projection.presentation import ComparisonRenderOptions
from cryosparc_2d_projection.axis_search import AxisProximityConfig, AxisSearchConfig
from cryosparc_2d_projection.surface_render import ClassRenderOptions


def build_parser():
    parser = argparse.ArgumentParser(description="Rank 2D classes by symmetry axis.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--select-job", required=True)
    parser.add_argument("--select-output", default="templates_selected")
    parser.add_argument("--volume-job", required=True)
    parser.add_argument("--volume-output", default="volume")
    families = parser.add_mutually_exclusive_group()
    families.add_argument("--axis-family", choices=("2fold", "3fold", "5fold"))
    families.add_argument("--axis-families")
    parser.add_argument("--low-resolution-A", type=float, default=80.0)
    parser.add_argument("--high-resolution-A", type=float, default=15.0)
    parser.add_argument("--mask-radius-fraction", type=float, default=0.45)
    parser.add_argument("--mask-edge-fraction", type=float, default=0.05)
    parser.add_argument("--roll-coarse-step", type=float, default=5.0)
    parser.add_argument("--roll-refine-step", type=float, default=0.5)
    parser.add_argument("--shift-bound-fraction", type=float, default=0.10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--mirror-warning-margin", type=float, default=0.05)
    parser.add_argument("--axis-cone-degrees", type=float, default=15.0)
    parser.add_argument("--tilt-coarse-step", type=float, default=3.0)
    parser.add_argument("--tilt-refine-step", type=float, default=0.5)
    parser.add_argument("--axis-roll", action="append", default=[])
    parser.add_argument("--comparison-dpi", type=int, default=100)
    parser.add_argument("--preview-page-size", type=int, default=10)
    parser.add_argument("--render-map", choices=("map", "sharpened"), default="map")
    parser.add_argument("--render-background", choices=("dark", "light"), default="dark")
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--render-grid-size", type=int)
    parser.add_argument("--surface-level", type=float)
    return parser


def main(argv=None, *, client_factory=None):
    args = build_parser().parse_args(argv)
    config = AxisSearchConfig(
        low_resolution_A=args.low_resolution_A,
        high_resolution_A=args.high_resolution_A,
        mask_radius_fraction=args.mask_radius_fraction,
        mask_edge_fraction=args.mask_edge_fraction,
        roll_coarse_step_degrees=args.roll_coarse_step,
        roll_refine_step_degrees=args.roll_refine_step,
        shift_bound_fraction=args.shift_bound_fraction,
        top_n=args.top_n,
        mirror_warning_margin=args.mirror_warning_margin,
    )
    if client_factory is None:
        from cryosparc.tools import CryoSPARC

        client_factory = CryoSPARC
    client = client_factory(args.url)
    if not client.test_connection():
        raise ConnectionError(f"Could not connect to CryoSPARC at {args.url}")
    project = client.find_project(args.project)
    run_axis_search_job(
        project,
        args.workspace,
        AxisSourceOutput(args.select_job, args.select_output),
        AxisSourceOutput(args.volume_job, args.volume_output),
        families=_parse_axis_families(args.axis_family, args.axis_families),
        config=config,
        proximity_config=AxisProximityConfig(
            cone_degrees=args.axis_cone_degrees,
            coarse_step_degrees=args.tilt_coarse_step,
            refine_step_degrees=args.tilt_refine_step,
        ),
        axis_rolls=parse_axis_rolls(args.axis_roll),
        comparison_options=ComparisonRenderOptions(
            dpi=args.comparison_dpi,
            page_size=args.preview_page_size,
        ),
        render_options=ClassRenderOptions(
            surface_level=args.surface_level,
            map_name=args.render_map,
            background=args.render_background,
            image_size=args.render_size,
            grid_size=args.render_grid_size,
        ),
    )
    return 0


def _parse_axis_families(single, multiple):
    if single is not None:
        return (single,)
    if multiple is None:
        return None
    values = tuple(item.strip() for item in multiple.split(",") if item.strip())
    if not values:
        raise ValueError("axis families must not be empty")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
