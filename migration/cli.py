from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import Nd2Selection, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migration-track",
        description="Segment and track one ND2 position/channel/z plane across all frames.",
    )
    parser.add_argument("nd2_path", type=Path, help="Path to the ND2 file.")
    parser.add_argument("--position", type=int, required=True, help="Zero-based ND2 position index.")
    parser.add_argument("--channel", type=int, required=True, help="Zero-based ND2 channel index.")
    parser.add_argument("--z", type=int, required=True, help="Zero-based ND2 z-slice index.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for the overlay PNG and trajectories CSV.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device for Cellpose and Trackastra.",
    )
    parser.add_argument("--diameter", type=float, help="Optional Cellpose diameter hint in pixels.")
    parser.add_argument(
        "--tracking-mode",
        choices=["greedy", "greedy_nodiv"],
        default="greedy",
        help="Trackastra linking mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.diameter is not None and args.diameter <= 0:
        parser.error("--diameter must be greater than 0")

    try:
        outputs = run_pipeline(
            nd2_path=args.nd2_path,
            selection=Nd2Selection(position=args.position, channel=args.channel, z=args.z),
            out_dir=args.out_dir,
            device_name=args.device,
            diameter=args.diameter,
            tracking_mode=args.tracking_mode,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Overlay: {outputs.overlay_path}")
    print(f"Trajectories: {outputs.trajectories_path}")
    print(f"Rows: {outputs.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
