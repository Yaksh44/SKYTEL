\
import os
import cv2
import time
import yaml
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from dataclasses import dataclass
from typing import Dict, Tuple
import subprocess
from utils import denorm_pts, point_in_polygon, draw_translucent_poly, put_text

YT_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"

@dataclass
class TrackInfo:
    last_centroid: Tuple[int,int]
    last_frame: int
    counted: bool
    lane_id: int  # 1,2,3 or -1 unknown
    last_y: float

def download_youtube(url: str, out_path: str) -> str:
    """
    Download a YouTube video to out_path using yt-dlp. Returns the file path.
    """
    os.makedirs(out_path, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", os.path.join(out_path, "%(title).80s.%(ext)s"),
        url
    ]
    print("Downloading video with yt-dlp...")
    subprocess.run(cmd, check=True)
    # pick largest mp4 in folder
    cands = [os.path.join(out_path, f) for f in os.listdir(out_path) if f.lower().endswith(".mp4")]
    if not cands:
        raise RuntimeError("No MP4 downloaded; check yt-dlp/install.")
    cands.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return cands[0]

def assign_lane(centroid, lane_polys):
    for idx, poly in enumerate(lane_polys, start=1):
        if point_in_polygon(centroid, poly):
            return idx
    return -1

def main(
    source: str = None,
    output_dir: str = "outputs",
    cfg_path: str = "config.yaml",
    model_name: str = "yolov8n.pt",
    show: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Prepare source
    if source is None:
        video_path = download_youtube(YT_URL, out_path=os.path.join(output_dir, "downloads"))
    else:
        video_path = source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    out_w = cfg.get("output_width") or width
    out_h = cfg.get("output_height") or height
    out_fps = cfg.get("output_fps") or fps

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = os.path.join(output_dir, "overlay_output.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (out_w, out_h))

    # Prepare YOLO
    model = YOLO(model_name)
    conf_th = cfg.get("conf_threshold", 0.3)
    vehicle_classes = set(cfg.get("vehicle_classes", [2,3,5,7]))

    # Prepare DeepSORT
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3, nn_budget=None)

    # Lane polygons
    lane_defs = cfg["lanes"]
    lane_polys = [denorm_pts(ld["polygon"], width, height) for ld in lane_defs]

    # Counting line
    cy = int(cfg.get("counting_line_y", 0.7) * height)
    direction = cfg.get("direction", "down")  # 'down','up','both'

    # Data structures
    tracks: Dict[int, TrackInfo] = {}
    counted_ids_per_lane = {1:set(), 2:set(), 3:set()}
    csv_rows = []

    pbar = tqdm(total=total_frames, disable=(total_frames is None), desc="Processing")
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts = frame_idx / fps if fps else 0.0

        # Inference
        results = model.predict(frame, conf=conf_th, verbose=False)[0]
        dets = []
        for *xyxy, conf, cls in results.boxes.data.cpu().numpy():
            cls = int(cls)
            if cls in vehicle_classes:
                x1,y1,x2,y2 = map(int, xyxy)
                dets.append(([x1, y1, x2-x1, y2-y1], float(conf), cls))

        # Update tracker
        tracks_out = tracker.update_tracks(raw_detections=dets, frame=frame)


        # Draw lanes
        for i, poly in enumerate(lane_polys, start=1):
            draw_translucent_poly(frame, poly, color=(0, 255 if i==1 else 200, 0), alpha=0.15, thickness=2)

        # Draw counting line
        cv2.line(frame, (0, cy), (width, cy), (0, 255, 255), 2)
        put_text(frame, "Counting line", (10, cy-10), scale=0.6, color=(0,0,0), thickness=2, bg=True)

        # Process tracks
        for t in tracks_out:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            tid = int(t.track_id)
            ltrb = t.to_ltrb()
            x1,y1,x2,y2 = map(int, ltrb)
            cx = int((x1 + x2) / 2)
            cy_obj = int((y1 + y2) / 2)
            centroid = (cx, cy_obj)

            # Previous info
            prev_info = tracks.get(tid)
            prev_y = prev_info.last_y if prev_info else cy_obj

            # Assign lane
            lane_id = assign_lane(centroid, lane_polys)

            # Update TrackInfo
            info = TrackInfo(last_centroid=centroid, last_frame=frame_idx, counted=False, lane_id=lane_id, last_y=cy_obj)
            tracks[tid] = info

            # Draw bbox/label
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 50, 50), 2)
            label = f"ID {tid} L{info.lane_id if info.lane_id!=-1 else '?'}"
            put_text(frame, label, (x1, max(20,y1-8)), scale=0.6, color=(255,255,255), thickness=2, bg=True)
            cv2.circle(frame, centroid, 3, (0,0,255), -1)

            # Counting logic
            crossed = False
            if info.lane_id in (1,2,3):
                if direction == "down" and prev_y < cy and cy_obj >= cy:
                    crossed = True
                elif direction == "up" and prev_y > cy and cy_obj <= cy:
                    crossed = True
                elif direction == "both" and ((prev_y < cy and cy_obj >= cy) or (prev_y > cy and cy_obj <= cy)):
                    crossed = True

            if crossed and (tid not in counted_ids_per_lane[info.lane_id]):
                counted_ids_per_lane[info.lane_id].add(tid)
                csv_rows.append({
                    "vehicle_id": tid,
                    "lane": info.lane_id,
                    "frame": frame_idx,
                    "timestamp_sec": round(ts, 3)
                })

        # Overlay counts
        y0 = 28
        for i in [1,2,3]:
            put_text(frame, f"{lane_defs[i-1]['name']}: {len(counted_ids_per_lane[i])}", (10, y0), scale=0.8, color=(255,255,255), thickness=2, bg=True)
            y0 += 28

        writer.write(frame)
        if show:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # Save CSV
    df = pd.DataFrame(csv_rows, columns=["vehicle_id","lane","frame","timestamp_sec"])
    csv_path = os.path.join(output_dir, "counts.csv")
    df.to_csv(csv_path, index=False)

    # Summary
    summary_txt = os.path.join(output_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        for i in [1,2,3]:
            f.write(f"Lane {i} total: {len(counted_ids_per_lane[i])}\n")

    print(f"Saved overlay video: {out_video_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_txt}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Traffic Flow Analysis - 3 Lane Vehicle Counter")
    ap.add_argument("--source", type=str, default=None, help="Path to local video file. If omitted, YouTube video will be downloaded.")
    ap.add_argument("--output_dir", type=str, default="outputs", help="Directory to store outputs")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights (e.g., yolov8n.pt)")
    ap.add_argument("--show", action="store_true", help="Display live window")
    args = ap.parse_args()
    main(source=args.source, output_dir=args.output_dir, cfg_path=args.config, model_name=args.model, show=args.show)
