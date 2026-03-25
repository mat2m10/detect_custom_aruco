"""
extract_pose.py
---------------
Extract ArUco marker pose (rotation angles) from all frames in a folder.
Saves results to a CSV for analysis and use in stabilization.

Usage:
    python extract_pose.py --frames data/mov/test_frames/ --params camera_params.npz
    python extract_pose.py --frames data/mov/test_frames/ --params camera_params.npz --plot
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import glob
import os
import csv

DICT_TYPE      = aruco.DICT_4X4_50
MARKER_SIZE_MM = 30.0
MIN_MARKER_PX  = 50  # ignore detections smaller than this (noise filter)


def preprocess(gray):
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray   = clahe.apply(gray)
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    return cv2.filter2D(gray, -1, kernel)


def make_detector():
    dictionary = aruco.getPredefinedDictionary(DICT_TYPE)
    params     = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin    = 3
    params.adaptiveThreshWinSizeMax    = 53
    params.adaptiveThreshWinSizeStep   = 10
    params.minMarkerPerimeterRate      = 0.02
    params.polygonalApproxAccuracyRate = 0.08
    return aruco.ArucoDetector(dictionary, params)


def rvec_to_euler(rvec):
    """Convert Rodrigues rotation vector to Euler angles (degrees) XYZ."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return rx, ry, rz


def marker_size_px(corner):
    c = corner[0]
    sides = [
        np.linalg.norm(c[1] - c[0]),
        np.linalg.norm(c[2] - c[1]),
        np.linalg.norm(c[3] - c[2]),
        np.linalg.norm(c[0] - c[3]),
    ]
    return np.mean(sides)


def process(frames_dir, params_path, do_plot):
    # Load calibration
    params        = np.load(params_path)
    camera_matrix = params["camera_matrix"].copy()
    dist_coeffs   = params["dist_coeffs"]

    detector = make_detector()

    # Collect frames
    exts   = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    frames = []
    for ext in exts:
        for f in glob.glob(os.path.join(frames_dir, ext)):
            if "_test" not in f and "_pose" not in f:
                frames.append(f)
    frames = sorted(frames)

    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Processing {len(frames)} frames...")

    rows        = []
    detected    = 0
    prev_rx     = None

    for fpath in frames:
        fname = os.path.basename(fpath)
        # Extract frame number from filename
        frame_num = int(''.join(filter(str.isdigit, fname.split('.')[0])))

        img = cv2.imread(fpath)
        if img is None:
            continue

        # Downscale
        h, w  = img.shape[:2]
        scale = 1.0
        if w > 1280:
            scale         = 1280 / w
            img           = cv2.resize(img, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
            cm            = camera_matrix.copy()
            cm[0]        *= scale
            cm[1]        *= scale
        else:
            cm = camera_matrix

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = preprocess(gray)

        corners, ids, _ = detector.detectMarkers(gray)

        best = None  # pick largest valid marker if multiple detected

        if ids is not None:
            for corner, mid in zip(corners, ids.ravel()):
                sz = marker_size_px(corner)
                if sz < MIN_MARKER_PX:
                    continue  # skip tiny false positives
                if best is None or sz > best["size"]:
                    best = {"corner": corner, "id": int(mid), "size": sz}

        if best is not None:
            # estimatePoseSingleMarkers removed in newer OpenCV — use solvePnP
            half = MARKER_SIZE_MM / 2.0
            obj_pts = np.array([[-half,  half, 0],
                                 [ half,  half, 0],
                                 [ half, -half, 0],
                                 [-half, -half, 0]], dtype=np.float32)
            img_pts = best["corner"][0].astype(np.float32)
            _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cm, dist_coeffs)
            rx, ry, rz = rvec_to_euler(rvec)
            prev_rx     = rx
            detected   += 1

            rows.append({
                "frame":    frame_num,
                "file":     fname,
                "detected": 1,
                "id":       best["id"],
                "size_px":  round(best["size"], 1),
                "rx":       round(rx, 3),
                "ry":       round(ry, 3),
                "rz":       round(rz, 3),
                "tx":       round(float(tvec[0][0]), 3),
                "ty":       round(float(tvec[1][0]), 3),
                "tz":       round(float(tvec[2][0]), 3),
            })
        else:
            rows.append({
                "frame":    frame_num,
                "file":     fname,
                "detected": 0,
                "id":       "",
                "size_px":  "",
                "rx":       "",
                "ry":       "",
                "rz":       "",
                "tx":       "",
                "ty":       "",
                "tz":       "",
            })

    # Save CSV
    csv_path = os.path.join(frames_dir, "pose_data.csv")
    fieldnames = ["frame", "file", "detected", "id", "size_px",
                  "rx", "ry", "rz", "tx", "ty", "tz"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetected: {detected}/{len(frames)} frames")
    print(f"Saved:    {csv_path}")

    # Print Rx stats for detected frames
    rx_vals = [r["rx"] for r in rows if r["detected"] == 1]
    if rx_vals:
        print(f"\nRx (tilt) statistics across detected frames:")
        print(f"  Mean:  {np.mean(rx_vals):+.2f}°")
        print(f"  Std:   {np.std(rx_vals):.2f}°")
        print(f"  Min:   {np.min(rx_vals):+.2f}°")
        print(f"  Max:   {np.max(rx_vals):+.2f}°")
        print(f"  Range: {np.max(rx_vals) - np.min(rx_vals):.2f}°")
        print()
        print("If Std is high, you have a lot of tilt wobble to correct.")
        print("If Mean is consistently offset (e.g. always ~+15°), "
              "your marker is mounted at an angle.")

    # Optional plot
    if do_plot:
        try:
            import matplotlib.pyplot as plt

            det_frames = [r["frame"] for r in rows if r["detected"] == 1]
            rx_vals    = [r["rx"]    for r in rows if r["detected"] == 1]
            ry_vals    = [r["ry"]    for r in rows if r["detected"] == 1]
            rz_vals    = [r["rz"]    for r in rows if r["detected"] == 1]

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            fig.suptitle("Marker rotation over time")

            axes[0].plot(det_frames, rx_vals, color="red")
            axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
            axes[0].set_ylabel("Rx (tilt) °")
            axes[0].set_title("X rotation — this is what we stabilize away")

            axes[1].plot(det_frames, ry_vals, color="green")
            axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
            axes[1].set_ylabel("Ry (top/bottom) °")

            axes[2].plot(det_frames, rz_vals, color="blue")
            axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
            axes[2].set_ylabel("Rz (turntable) °")
            axes[2].set_xlabel("Frame number")

            plt.tight_layout()
            plot_path = os.path.join(frames_dir, "pose_plot.png")
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved: {plot_path}")
            plt.show()

        except ImportError:
            print("matplotlib not installed — skipping plot.")
            print("Install with: pip install matplotlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True,
                        help="Folder containing extracted frames")
    parser.add_argument("--params", default="camera_params.npz",
                        help="Camera calibration file")
    parser.add_argument("--plot",   action="store_true",
                        help="Generate a plot of rotation over time")
    args = parser.parse_args()
    process(args.frames, args.params, args.plot)