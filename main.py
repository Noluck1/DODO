import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


EMPTY = "table_empty"
OCCUPIED = "table_occupied"
APPROACH = "approach"


@dataclass
class Event:
    frame_idx: int
    seconds: float
    event: str


class TableStateMachine:
    """Tracks table occupancy with stabilization and logs requested events."""

    def __init__(
        self,
        fps: float,
        min_stable_seconds: float = 1.0,
        min_empty_before_approach: float = 1.5,
    ) -> None:
        self.fps = fps if fps > 0 else 25.0
        self.min_stable_frames = max(1, int(min_stable_seconds * self.fps))
        self.min_empty_before_approach = min_empty_before_approach

        self.stable_state: Optional[bool] = None  # False=empty, True=occupied
        self.pending_state: Optional[bool] = None
        self.pending_count: int = 0

        self.events: List[Event] = []
        self.last_empty_time: Optional[float] = None

    def _push_event(self, frame_idx: int, seconds: float, event: str) -> None:
        self.events.append(Event(frame_idx=frame_idx, seconds=seconds, event=event))

    def update(self, occupied_raw: bool, frame_idx: int, seconds: float) -> bool:
        """Update by raw occupancy and return stabilized occupancy state."""
        if self.stable_state is None:
            self.stable_state = occupied_raw
            if occupied_raw:
                self._push_event(frame_idx, seconds, OCCUPIED)
            else:
                self._push_event(frame_idx, seconds, EMPTY)
                self.last_empty_time = seconds
            return self.stable_state

        if occupied_raw == self.stable_state:
            self.pending_state = None
            self.pending_count = 0
            return self.stable_state

        if self.pending_state != occupied_raw:
            self.pending_state = occupied_raw
            self.pending_count = 1
            return self.stable_state

        self.pending_count += 1
        if self.pending_count < self.min_stable_frames:
            return self.stable_state

        previous_state = self.stable_state
        self.stable_state = occupied_raw
        self.pending_state = None
        self.pending_count = 0

        if self.stable_state:
            if previous_state is False and self.last_empty_time is not None:
                if (seconds - self.last_empty_time) >= self.min_empty_before_approach:
                    self._push_event(frame_idx, seconds, APPROACH)
            self._push_event(frame_idx, seconds, OCCUPIED)
        else:
            self._push_event(frame_idx, seconds, EMPTY)
            self.last_empty_time = seconds

        return self.stable_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table cleaning detection prototype")
    parser.add_argument(
        "--video",
        help="Path to source video. If omitted, the first *.mp4 in current folder is used",
    )
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--events_csv", default="events.csv", help="Events table path")
    parser.add_argument("--report_txt", default="report.txt", help="Text report path")

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Fixed ROI coordinates; if omitted, ROI will be selected interactively",
    )

    parser.add_argument("--fg_history", type=int, default=300)
    parser.add_argument("--fg_var_threshold", type=float, default=25.0)
    parser.add_argument("--min_contour_area", type=int, default=600)
    parser.add_argument("--occupied_ratio", type=float, default=0.012)
    parser.add_argument("--min_stable_seconds", type=float, default=1.0)
    parser.add_argument("--min_empty_before_approach", type=float, default=1.5)
    parser.add_argument("--display", action="store_true", help="Show processing window")
    parser.add_argument(
        "--progress_every_sec",
        type=float,
        default=5.0,
        help="How often to print processing progress (in seconds)",
    )
    return parser.parse_args()


def compute_intersection_area(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def select_or_validate_roi(
    frame: np.ndarray, roi_arg: Optional[List[int]]
) -> Tuple[int, int, int, int]:
    if roi_arg is not None:
        x, y, w, h = roi_arg
        return int(x), int(y), int(w), int(h)

    window_name = "Select table ROI (drag mouse, Enter=confirm, C=cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {
        "drawing": False,
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0,
        "has_rect": False,
    }

    def on_mouse(event: int, mx: int, my: int, flags: int, param: object) -> None:
        _ = flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["x0"], state["y0"] = mx, my
            state["x1"], state["y1"] = mx, my
            state["has_rect"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["x1"], state["y1"] = mx, my
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            state["x1"], state["y1"] = mx, my
            state["has_rect"] = True

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = frame.copy()
        if state["drawing"] or state["has_rect"]:
            x0, y0 = state["x0"], state["y0"]
            x1, y1 = state["x1"], state["y1"]
            cv2.rectangle(
                canvas,
                (min(x0, x1), min(y0, y1)),
                (max(x0, x1), max(y0, y1)),
                (0, 255, 255),
                2,
            )
        cv2.putText(
            canvas,
            "Drag ROI, Enter=confirm, C=cancel",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter/Space
            if not state["has_rect"] and not state["drawing"]:
                continue
            x0, y0 = state["x0"], state["y0"]
            x1, y1 = state["x1"], state["y1"]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            if w <= 0 or h <= 0:
                raise ValueError("ROI not selected or invalid.")
            return int(x), int(y), int(w), int(h)
        if key in (ord("c"), 27):  # C/Esc
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            raise ValueError("ROI selection cancelled.")


def build_delays(events_df: pd.DataFrame) -> pd.DataFrame:
    empties = events_df[events_df["event"] == EMPTY][["seconds"]].copy()
    approaches = events_df[events_df["event"] == APPROACH][["seconds"]].copy()

    if empties.empty or approaches.empty:
        return pd.DataFrame(columns=["empty_at", "approach_at", "delay_seconds"])

    delay_rows = []
    approach_times = approaches["seconds"].to_list()
    used_idx = 0

    for empty_time in empties["seconds"].to_list():
        while used_idx < len(approach_times) and approach_times[used_idx] <= empty_time:
            used_idx += 1
        if used_idx >= len(approach_times):
            break
        approach_time = approach_times[used_idx]
        delay_rows.append(
            {
                "empty_at": float(empty_time),
                "approach_at": float(approach_time),
                "delay_seconds": float(approach_time - empty_time),
            }
        )
        used_idx += 1

    return pd.DataFrame(delay_rows)


def save_report(report_path: Path, delays_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    if delays_df.empty:
        avg_delay = None
    else:
        avg_delay = float(delays_df["delay_seconds"].mean())

    lines = [
        "Table occupancy report",
        "====================",
        f"Total events: {len(events_df)}",
        f"Empty events: {(events_df['event'] == EMPTY).sum()}",
        f"Occupied events: {(events_df['event'] == OCCUPIED).sum()}",
        f"Approach events: {(events_df['event'] == APPROACH).sum()}",
    ]

    if avg_delay is None:
        lines.append("Average delay (empty -> next approach): n/a")
    else:
        lines.append(f"Average delay (empty -> next approach): {avg_delay:.2f} sec")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        candidates = sorted(Path.cwd().glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(
                "Video file is not provided. Use --video <path> or place an .mp4 file in current folder."
            )
        video_path = candidates[0]
        print(f"--video is not set, using: {video_path.resolve()}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Cannot read first frame from video.")

    print("Preparing ROI selection...")
    x, y, w, h = select_or_validate_roi(first_frame, args.roi)
    print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
    roi_rect = (x, y, x + w, y + h)

    writer = cv2.VideoWriter(
        str(Path(args.output)),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25.0,
        (width, height),
    )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.fg_history,
        varThreshold=args.fg_var_threshold,
        detectShadows=False,
    )

    state = TableStateMachine(
        fps=fps,
        min_stable_seconds=args.min_stable_seconds,
        min_empty_before_approach=args.min_empty_before_approach,
    )

    frame_idx = 0
    last_progress_print_sec = -1e9
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        seconds = frame_idx / (fps if fps > 0 else 25.0)
        if (seconds - last_progress_print_sec) >= max(0.5, args.progress_every_sec):
            if total_frames > 0:
                pct = (frame_idx / total_frames) * 100.0
                print(f"Processing: frame {frame_idx}/{total_frames} ({pct:.1f}%)")
            else:
                print(f"Processing: frame {frame_idx}")
            last_progress_print_sec = seconds

        fgmask = subtractor.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = w * h
        motion_area_in_roi = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_contour_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            contour_rect = (bx, by, bx + bw, by + bh)
            motion_area_in_roi += compute_intersection_area(roi_rect, contour_rect)

        occupied_raw = (motion_area_in_roi / max(1, roi_area)) >= args.occupied_ratio
        occupied = state.update(occupied_raw=occupied_raw, frame_idx=frame_idx, seconds=seconds)

        color = (0, 0, 255) if occupied else (0, 200, 0)
        label = "OCCUPIED" if occupied else "EMPTY"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label} | motion_ratio={motion_area_in_roi / max(1, roi_area):.3f}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if args.display:
            cv2.imshow("table-tracker", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    events_df = pd.DataFrame([e.__dict__ for e in state.events])
    if events_df.empty:
        events_df = pd.DataFrame(columns=["frame_idx", "seconds", "event"])

    events_df.to_csv(args.events_csv, index=False)
    delays_df = build_delays(events_df)
    delays_path = Path(args.events_csv).with_name("delays.csv")
    delays_df.to_csv(delays_path, index=False)

    save_report(Path(args.report_txt), delays_df, events_df)

    if delays_df.empty:
        print("Average delay (empty -> next approach): n/a")
    else:
        avg_delay = delays_df["delay_seconds"].mean()
        print(f"Average delay (empty -> next approach): {avg_delay:.2f} sec")

    print(f"Output video saved to: {Path(args.output).resolve()}")
    print(f"Events CSV saved to: {Path(args.events_csv).resolve()}")
    print(f"Delays CSV saved to: {delays_path.resolve()}")
    print(f"Report saved to: {Path(args.report_txt).resolve()}")


if __name__ == "__main__":
    main()
