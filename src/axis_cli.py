"""Command-line entry point for image-only Symmetry-Axis Class Search."""

import argparse
import sys

from cryosparc_2d_projection.axis_external_job import run_axis_search_job
from cryosparc_2d_projection.axis_presentation import parse_axis_rolls
from cryosparc_2d_projection.axis_registry import AxisFamilyRegistry
from cryosparc_2d_projection.cli import parse_supported_symmetry


class _AxisArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            registry = AxisFamilyRegistry.for_symmetry(parsed.symmetry)
            if parsed.axis_family is not None:
                parsed.axis_family = tuple(registry.lookup(name).name for name in parsed.axis_family)
            parse_axis_rolls(parsed.axis_roll, symmetry=parsed.symmetry)
        except ValueError as error:
            self.error(str(error))
        return parsed


def build_parser():
    parser = _AxisArgumentParser(description="Rank 2D classes by symmetry axis.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--select-job", required=True)
    parser.add_argument("--select-output", default="templates_selected")
    parser.add_argument("--volume-job", required=True)
    parser.add_argument("--volume-output", default="volume")
    parser.add_argument(
        "--symmetry", type=parse_supported_symmetry, default="I",
        help="Map symmetry: Cn (n >= 2), Dn, T, O, I (default: I)",
    )
    parser.add_argument(
        "--axis-family",
        type=_parse_axis_families,
        metavar="FAMILY[,FAMILY...]",
        help="Axis families for the selected symmetry, e.g. 4fold or 3fold-2 (default: all)",
    )
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
    parser.add_argument(
        "--refine-near-axis",
        action="store_true",
        help="Refine selected Exact-Axis candidates inside the configured cone.",
    )
    parser.add_argument("--axis-roll", action="append", default=[])
    parser.add_argument("--comparison-dpi", type=int, default=100)
    parser.add_argument("--preview-page-size", type=int, default=10)
    parser.add_argument("--render-map", choices=("map", "sharpened"), default="map")
    parser.add_argument("--render-background", choices=("dark", "light"), default="dark")
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--render-grid-size", type=int)
    parser.add_argument("--surface-level", type=float)
    parser.add_argument(
        "--auto-crop-2d",
        action="store_true",
        help=(
            "Automatically crop 2D Class Average and Matched Projection panels "
            "in comparison previews"
        ),
    )
    return parser


def main(argv=None, *, client_factory=None):
    from cryosparc_2d_projection.workflow_config import prepare_workflow

    prepared = prepare_workflow('axis', argv)
    if client_factory is None:
        from cryosparc.tools import CryoSPARC

        client_factory = CryoSPARC
    client = client_factory(prepared.url)
    if not client.test_connection():
        raise ConnectionError(f"Could not connect to CryoSPARC at {prepared.url}")
    project = client.find_project(prepared.project)
    run_axis_search_job(
        project,
        **prepared.options,
        status_callback=print,
        warning_callback=lambda message: print(message, file=sys.stderr),
    )
    return 0


def _parse_axis_families(value):
    if value is None:
        return None

    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("axis family list must not be empty")
    # Registry validation happens after all options, independent of their order.
    return values


if __name__ == "__main__":
    raise SystemExit(main())
