"""
correct_frames.py
-----------------
Rotate each frame to correct for the X-axis tilt using the interpolated
pose data from interpolate_pose.py.

The mean Rx across all detected frames is used as the "baseline" angle
(your marker's resting mount angle). Each frame is corrected relative
to that baseline so the output looks level.

Usage:
    python correct_frames.py --frames data/mov/test_frames/ \
                             --csv data/mov/test_frames/pose_data_interpolated.csv \
                             --params camera_params.npz
"""

import cv2
import numpy as np
import csv
import argparse
import os
import glob


def load_pose_csv(csv_path):
    """Load interpolated pose CSV into a dict keyed by frame number."""
    poses = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            rx    = float(row["rx"]) if row["rx"] else None
            ry    = float(row["ry"]) if row["ry"] else None
            rz    = float(row["rz"]) if row["rz"] else None
            poses[frame] = {
                "rx":       rx,
                "ry":       ry,
                "rz":       rz,
                "detected": int(row["detected"]),
                "spike":    row.get("spike", "False") == "True",
                "interp":   row.get("interp", "detected"),
            }
    return poses


def compute_baseline_rx(poses):
    """
    The marker is mounted at a fixed angle on the handle.
    Compute the median Rx across all clean detected frames as the baseline.
    We use median (not mean) to be robust against remaining outliers.
    """
    rx_vals = [
        p["rx"] for p in poses.values()
        if p["detected"] == 1 and not p["spike"] and p["rx"] is not None
    ]
    if not rx_vals:
        return 0.0
    return float(np.median(rx_vals))


def rotate_frame(frame, angle_deg, camera_matrix):
    """
    Rotate a frame by angle_deg around its center.
    Uses the principal point from camera_matrix as the rotation center
    (more accurate than image center).
    """
    h, w  = frame.shape[:2]
    cx    = camera_matrix[0, 2]
    cy    = camera_matrix[1, 2]

    # Clamp center to image bounds
    cx = float(np.clip(cx, 0, w))
    cy = float(np.clip(cy, 0, h))

    M   = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    out = cv2.warpAffine(frame, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out


def process(frames_dir, csv_path, params_path):
    # Load calibration
    params        = np.load(params_path)
    camera_matrix = params["camera_matrix"].copy()

    # Load pose data
    poses = load_pose_csv(csv_path)
    print(f"Loaded pose data for {len(poses)} frames")

    # Compute baseline (resting mount angle of marker)
    baseline_rx = compute_baseline_rx(poses)
    print(f"Baseline Rx (marker mount angle): {baseline_rx:.2f}°")
    print(f"This offset will be subtracted from every frame.")

    # Collect frames
    exts   = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    frames = []
    for ext in exts:
        for f in glob.glob(os.path.join(frames_dir, ext)):
            if "_test" not in f and "_pose" not in f and "_corrected" not in f:
                frames.append(f)
    frames = sorted(frames)

    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    # Output folder
    out_dir = os.path.join(frames_dir, "corrected")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nProcessing {len(frames)} frames → {out_dir}/")

    # Scale camera matrix to match detection downscale
    # (frames were processed at 1280px wide)
    sample     = cv2.imread(frames[0])
    h, w       = sample.shape[:2]
    scale      = 1.0
    if w > 1280:
        scale  = 1280 / w
        cm     = camera_matrix.copy()
        cm[0] *= scale
        cm[1] *= scale
    else:
        cm = camera_matrix

    skipped = 0
    corrected = 0

    for fpath in frames:
        fname     = os.path.basename(fpath)
        digits = ''.join(filter(str.isdigit, fname.split('.')[0]))
        if not digits:
            continue
        frame_num = int(digits)

        img = cv2.imread(fpath)
        if img is None:
            continue

        # Downscale to match pose estimation scale
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        pose = poses.get(frame_num)

        if pose is None or pose["rx"] is None or pose["spike"]:
            # No pose data — save frame uncorrected
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, img)
            skipped += 1
            continue

        # Correction angle = deviation from baseline
        correction = pose["rx"] - baseline_rx

        # Annotate source
        source = pose["interp"] if pose["interp"] else "detected"
        color  = (0, 255, 0) if source == "detected" else (0, 165, 255)

        corrected_img = rotate_frame(img, -correction, cm)

        # Overlay info
        cv2.putText(corrected_img,
                    f"Rx={pose['rx']:+.1f}  base={baseline_rx:.1f}  "
                    f"corr={correction:+.1f}  [{source}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, corrected_img)
        corrected += 1

    print(f"Done.")
    print(f"  Corrected: {corrected} frames")
    print(f"  Skipped (no pose): {skipped} frames")
    print(f"  Output: {out_dir}/")
    print()
    print("Next step: reassemble frames into video with:")
    print(f"  ffmpeg -framerate 30 -pattern_type glob -i '{out_dir}/*.jpg' "
          f"-c:v libx264 -pix_fmt yuv420p corrected.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",  required=True,
                        help="Folder containing extracted frames")
    parser.add_argument("--csv",     required=True,
                        help="Interpolated pose CSV file")
    parser.add_argument("--params",  default="camera_params.npz",
                        help="Camera calibration file")
    args = parser.parse_args()
    process(args.frames, args.csv, args.params)