#!/usr/bin/env python3
"""Generate fig2 comparison SVGs for the LISCA paper (A–F column layout).

Produces fig2_a549.svg and fig2_mda231.svg using plot_roi_figure.py.

Run with:
    /home/jack/workspace/migration/.venv/bin/python \
        /home/jack/workspace/migration/scripts/plot_fig2_paper.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    data_dir = Path("/home/jack/data/lisca_review/fig2/20260519")
    out_dir = Path("/home/jack/workspace/lisca-paper/figs")
    script = Path(__file__).resolve().parent / "plot_roi_figure.py"
    python = Path("/home/jack/workspace/migration/.venv/bin/python")

    variants = [
        ("fig2_a549.svg", 26, 37, data_dir / "roi_cell_selection.json"),
        ("fig2_mda231.svg", 2, 44, data_dir / "roi_cell_selection_mda231.json"),
    ]

    for output_name, left_position, right_position, selection_path in variants:
        cmd = [
            str(python),
            str(script),
            str(data_dir),
            "--left-position",
            str(left_position),
            "--right-position",
            str(right_position),
            "--selection-json",
            str(selection_path),
            "--output",
            str(out_dir / output_name),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
