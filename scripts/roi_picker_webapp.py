#!/usr/bin/env python3
"""Web-based ROI and cell picker for the comparison figure."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from interactive_roi_picker import (  # noqa: E402
    PanelState,
    build_selection_payload,
    clamp_roi_origin,
    load_panel_state,
    panel_marker_payload,
    pick_track_at_click,
    render_panel_background_png,
)
from plot_roi_figure import _render_from_selection_payload  # noqa: E402

STATIC_DIR = SCRIPT_DIR / "static" / "roi_picker"
Side = Literal["left", "right"]


@dataclass
class PickerSession:
    data_dir: Path
    left: PanelState
    right: PanelState
    output_path: Path
    selection_json_path: Path
    left_position: int
    right_position: int
    channel: str
    z: int
    roi_fraction: float
    roi_size: int | None
    min_track_length: int
    inset_padding: int
    roi_rank: int
    cell_rank: int
    inner_margin_fraction: float
    inner_margin_px: int | None
    figure_output: Path | None
    background_cache: dict[Side, bytes]

    def panel_for_side(self, side: Side) -> PanelState:
        return self.left if side == "left" else self.right


class PickRequest(BaseModel):
    side: Side
    x: float
    y: float


class RoiUpdateRequest(BaseModel):
    side: Side
    roi_y: int
    roi_x: int


class SaveRequest(BaseModel):
    render: bool = True
    figure_output: str | None = None


class PanelStateResponse(BaseModel):
    position: int
    title: str
    width: int
    height: int
    roi_y: int
    roi_x: int
    roi_height: int
    roi_width: int
    track_ids: list[int]
    markers: list[dict]


class SessionStateResponse(BaseModel):
    data_dir: str
    output_path: str
    left: PanelStateResponse
    right: PanelStateResponse


class PickResponse(BaseModel):
    track_id: int | None
    track_ids: list[int]
    markers: list[dict]


class SaveResponse(BaseModel):
    selection_path: str
    figure_path: str | None = None


def panel_state_response(panel: PanelState, title: str) -> PanelStateResponse:
    height, width = panel.frame.shape
    return PanelStateResponse(
        position=panel.position,
        title=title,
        width=width,
        height=height,
        roi_y=panel.roi_y0,
        roi_x=panel.roi_x0,
        roi_height=panel.roi_height,
        roi_width=panel.roi_width,
        track_ids=list(panel.selected_ids),
        markers=panel_marker_payload(panel),
    )


def create_app(session: PickerSession) -> FastAPI:
    app = FastAPI(title="ROI Picker", docs_url="/api/docs", redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state", response_model=SessionStateResponse)
    def get_state() -> SessionStateResponse:
        return SessionStateResponse(
            data_dir=str(session.data_dir),
            output_path=str(session.selection_json_path),
            left=panel_state_response(session.left, "patterned"),
            right=panel_state_response(session.right, "unpatterned"),
        )

    @app.get("/api/background/{side}.png")
    def background_image(side: Side) -> Response:
        try:
            payload = session.background_cache[side]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown side: {side}") from exc
        return Response(content=payload, media_type="image/png")

    @app.post("/api/pick", response_model=PickResponse)
    def pick_cell(request: PickRequest) -> PickResponse:
        panel = session.panel_for_side(request.side)
        track_id = pick_track_at_click(panel, request.y, request.x)
        if track_id is not None:
            panel.selected_ids.append(track_id)
        return PickResponse(
            track_id=track_id,
            track_ids=list(panel.selected_ids),
            markers=panel_marker_payload(panel),
        )

    @app.patch("/api/roi", response_model=PanelStateResponse)
    def update_roi(request: RoiUpdateRequest) -> PanelStateResponse:
        panel = session.panel_for_side(request.side)
        panel.roi_y0, panel.roi_x0 = clamp_roi_origin(panel, request.roi_y, request.roi_x)
        title = "patterned" if request.side == "left" else "unpatterned"
        return panel_state_response(panel, title)

    @app.post("/api/save", response_model=SaveResponse)
    def save_selection(request: SaveRequest) -> SaveResponse:
        payload = build_selection_payload(session.data_dir, session.left, session.right)
        session.selection_json_path.parent.mkdir(parents=True, exist_ok=True)
        session.selection_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        figure_path: Path | None = None
        if request.render:
            output = (
                Path(request.figure_output).expanduser().resolve()
                if request.figure_output
                else session.figure_output
            )
            if output is None:
                raise HTTPException(status_code=400, detail="No figure output path configured")
            try:
                figure_path = _render_from_selection_payload(
                    payload,
                    data_dir=session.data_dir,
                    left_position=session.left_position,
                    right_position=session.right_position,
                    channel=session.channel,
                    z=session.z,
                    output_path=output,
                    roi_fraction=session.roi_fraction,
                    roi_size=session.roi_size,
                    min_track_length=session.min_track_length,
                    inset_padding=session.inset_padding,
                    roi_rank=session.roi_rank,
                    cell_rank=session.cell_rank,
                    inner_margin_fraction=session.inner_margin_fraction,
                    inner_margin_px=session.inner_margin_px,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return SaveResponse(
            selection_path=str(session.selection_json_path),
            figure_path=str(figure_path) if figure_path is not None else None,
        )

    return app


def build_session(
    *,
    data_dir: Path,
    left_position: int,
    right_position: int,
    channel: str,
    z: int,
    roi_fraction: float,
    roi_size: int | None,
    selection_json_path: Path,
    figure_output: Path | None,
    min_track_length: int,
    inset_padding: int,
    roi_rank: int,
    cell_rank: int,
    inner_margin_fraction: float,
    inner_margin_px: int | None,
) -> PickerSession:
    left = load_panel_state(data_dir, left_position, channel, z, roi_fraction, roi_size)
    right = load_panel_state(data_dir, right_position, channel, z, roi_fraction, roi_size)
    left.selected_ids = deque(maxlen=3)
    right.selected_ids = deque(maxlen=3)

    return PickerSession(
        data_dir=data_dir,
        left=left,
        right=right,
        output_path=selection_json_path,
        selection_json_path=selection_json_path,
        left_position=left_position,
        right_position=right_position,
        channel=channel,
        z=z,
        roi_fraction=roi_fraction,
        roi_size=roi_size,
        min_track_length=min_track_length,
        inset_padding=inset_padding,
        roi_rank=roi_rank,
        cell_rank=cell_rank,
        inner_margin_fraction=inner_margin_fraction,
        inner_margin_px=inner_margin_px,
        figure_output=figure_output,
        background_cache={
            "left": render_panel_background_png(left),
            "right": render_panel_background_png(right),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--left-position", type=int, default=26)
    parser.add_argument("--right-position", type=int, default=37)
    parser.add_argument("--channel", default="all")
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--roi-fraction", type=float, default=0.5)
    parser.add_argument("--roi-size", type=int, default=None)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="JSON path for saved ROI and cell selections",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=None,
        help="PNG path written when Save & render is used",
    )
    parser.add_argument("--min-track-length", type=int, default=50)
    parser.add_argument("--inset-padding", type=int, default=24)
    parser.add_argument("--roi-rank", type=int, default=1)
    parser.add_argument("--cell-rank", type=int, default=0)
    parser.add_argument("--roi-inner-margin-fraction", type=float, default=0.08)
    parser.add_argument("--roi-inner-margin-px", type=int, default=None)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    selection_json = args.selection_json or data_dir / "roi_cell_selection.json"
    figure_output = args.figure_output or data_dir / "roi_comparison.png"

    try:
        session = build_session(
            data_dir=data_dir,
            left_position=args.left_position,
            right_position=args.right_position,
            channel=args.channel,
            z=args.z,
            roi_fraction=args.roi_fraction,
            roi_size=args.roi_size,
            selection_json_path=selection_json.expanduser().resolve(),
            figure_output=figure_output.expanduser().resolve(),
            min_track_length=args.min_track_length,
            inset_padding=args.inset_padding,
            roi_rank=args.roi_rank,
            cell_rank=args.cell_rank,
            inner_margin_fraction=args.roi_inner_margin_fraction,
            inner_margin_px=args.roi_inner_margin_px,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    import uvicorn

    app = create_app(session)
    url = f"http://{args.host}:{args.port}/"
    print(f"ROI picker webapp at {url}")
    print(f"Selection JSON: {session.selection_json_path}")
    if not args.no_open_browser:
        import webbrowser

        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())