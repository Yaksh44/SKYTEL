# Traffic Flow Analysis – 3-Lane Vehicle Counting (YOLOv8 + DeepSORT)

A complete, ready-to-run pipeline that:
- Downloads a public traffic video from YouTube (or uses your own file)
- Detects vehicles with YOLOv8 (COCO)
- Tracks them with DeepSORT to avoid double counts
- Assigns each track to **one of three lanes** (polygon ROIs)
- Counts vehicles when they cross a horizontal counting line
- Exports a CSV and an overlaid MP4 with real-time lane counts
- Prints a lane-wise summary at the end

---
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run
```bash
python main.py --show
```

Outputs are written to `./outputs/`:
- `overlay_output.mp4` – annotated video (lanes, counts, track IDs)
- `counts.csv` – columns: `vehicle_id,lane,frame,timestamp_sec`
- `summary.txt` – lane-wise totals

---

## Lane Setup

Lanes are defined as polygons in **normalized coordinates** inside `config.yaml`.
Edit the points so they tightly cover the three lanes visible in your chosen camera angle.

- To tweak:
  1. Run once with `--show` and note where the polygons land.
  2. Update the polygons in `config.yaml` (values in range 0..1).
  3. Re-run.

The default config splits the frame into three vertical ROIs aimed at MNn9qKG2UFI (the demo link).

The **counting line** is a horizontal line at `counting_line_y` (0..1). A vehicle is counted for
its lane when its track **crosses** this line in the specified `direction` (`down`, `up`, or `both`).

---

## Accuracy Tips

- Try a larger YOLO model for accuracy (e.g., `--model yolov8s.pt`).
- Adjust `conf_threshold` in `config.yaml` if you see false positives/negatives.
- Ensure lane polygons don’t overlap; each centroid should fall into **at most one** lane.
- Place the counting line where traffic coherently flows past (e.g., lower third of frame).

---

## Performance Tips (Real-time/Near Real-time)

- Use `yolov8n.pt` for speed on CPU; switch to GPU for faster processing.
- Install GPU acceleration (CUDA) and PyTorch accordingly for your system.
- Lower the input resolution (set `output_width/height` in `config.yaml`) for speed.
- Set `--show` off when recording final outputs for maximum throughput.

---

## CSV Schema

| column         | type   | description                                  |
|----------------|--------|----------------------------------------------|
| vehicle_id     | int    | Persistent track ID from DeepSORT            |
| lane           | int    | 1..3 (as defined in `config.yaml`)           |
| frame          | int    | Frame index at the moment of counting        |
| timestamp_sec  | float  | Frame index / FPS                            |

---

---

## Technical Summary (Approach, Challenges, Solutions)

**Approach:** YOLOv8 (COCO) detects vehicles; DeepSORT maintains stable IDs. Each frame, we compute
track centroids and assign them to the first matching lane polygon; we count a track once when
it crosses the counting line in the configured direction. We export per-event rows to CSV and
overlay all visuals to MP4.

**Challenges & Solutions:**
- *Lane Assignment Ambiguity:* Overlapping or poorly shaped polygons can mis-assign tracks.
  → Provide normalized, easy-to-edit polygons and advice to avoid overlaps.
- *ID Switches in Heavy Traffic:* Short occlusions can cause re-ID issues.
  → DeepSORT with appearance embedding reduces switches; `max_age`/`n_init` tuned for stability.
- *Real-time on CPU:* Use `yolov8n` and optionally reduce frame size; disable display window for speed.
- *Counting Line Selection:* If placed too high/low, cross-traffic may be missed/overcounted.
  → Configurable `counting_line_y` and `direction` with guidance.

---
## License

MIT (or adapt to your needs)
