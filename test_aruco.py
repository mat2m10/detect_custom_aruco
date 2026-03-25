"""
test_aruco.py
-------------
Comprehensive detection test — gives detailed feedback on what was found,
what was almost found, and confidence metrics.

Useful for testing:
  - Multiple markers in frame
  - Bent / curved markers
  - Various angles
  - Different lighting conditions

Usage:
    # Test a single image
    python test_aruco.py --image photo.jpg

    # Test all images in a folder and get a summary
    python test_aruco.py --folder ./test_photos/

    # Live camera test (press Q to quit, S to save a snapshot)
    python test_aruco.py --live
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import glob
import os
from datetime import datetime


DICT_TYPE = aruco.DICT_4X4_50


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


def marker_warp_score(corner, gray):
    """
    Measure how much the detected marker region deviates from a flat square.
    Returns a score 0.0 (perfect square) to 1.0 (very bent/distorted).
    Useful for estimating how much the marker is bent.
    """
    pts = corner[0].astype(np.float32)

    # Ideal square side length = average of actual side lengths
    sides = [
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[1]),
        np.linalg.norm(pts[3] - pts[2]),
        np.linalg.norm(pts[0] - pts[3]),
    ]
    avg_side = np.mean(sides)
    side_variance = np.std(sides) / avg_side  # how unequal are the sides?

    # Diagonal ratio — a perfect square has diag1 == diag2
    d1 = np.linalg.norm(pts[2] - pts[0])
    d2 = np.linalg.norm(pts[3] - pts[1])
    diag_ratio = abs(d1 - d2) / max(d1, d2)

    score = (side_variance + diag_ratio) / 2
    return float(np.clip(score, 0, 1))


def analyze_frame(frame, detector, label=""):
    """
    Run detection on a frame and return rich analysis.
    Returns (annotated_frame, result_dict).
    """
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1280:
        scale = 1280 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_processed  = preprocess(gray)

    corners, ids, rejected = detector.detectMarkers(gray_processed)

    vis = frame.copy()
    result = {
        "label":    label,
        "found":    0,
        "ids":      [],
        "rejected": len(rejected),
        "markers":  [],
    }

    if ids is not None:
        aruco.drawDetectedMarkers(vis, corners, ids)
        result["found"] = len(ids)
        result["ids"]   = ids.ravel().tolist()

        for corner, mid in zip(corners, ids.ravel()):
            c      = corner[0]
            cx, cy = c.mean(axis=0).astype(int)
            score  = marker_warp_score(corner, gray_processed)

            # Compute apparent size in pixels
            sides = [
                np.linalg.norm(c[1] - c[0]),
                np.linalg.norm(c[2] - c[1]),
                np.linalg.norm(c[3] - c[2]),
                np.linalg.norm(c[0] - c[3]),
            ]
            avg_size = np.mean(sides)

            result["markers"].append({
                "id":        int(mid),
                "warp":      score,
                "size_px":   avg_size,
            })

            # Color warp score: green (flat) → yellow → red (very bent)
            r = int(255 * min(score * 4, 1.0))
            g = int(255 * (1 - min(score * 4, 1.0)))
            color = (0, g, r)

            cv2.putText(vis, f"ID {mid}", (cx - 20, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"warp={score:.2f}", (cx - 30, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw rejected candidates in red
    if rejected:
        rejected_wrapped = [r.reshape(1, 4, 2) for r in rejected]
        aruco.drawDetectedMarkers(vis, rejected_wrapped, None,
                                  borderColor=(0, 0, 255))

    # Draw summary overlay
    summary = (f"{label}  |  Found: {result['found']}  "
               f"Rejected: {result['rejected']}")
    cv2.putText(vis, summary, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, summary, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return vis, result


def print_result(result):
    label = result["label"] or "frame"
    if result["found"] > 0:
        print(f"  ✓ {label}")
        for m in result["markers"]:
            warp = m["warp"]
            warp_str = "flat" if warp < 0.05 else \
                       "slight bend" if warp < 0.15 else \
                       "moderate bend" if warp < 0.30 else "heavily bent"
            print(f"    ID {m['id']}  |  size={m['size_px']:.0f}px  "
                  f"|  warp={warp:.2f} ({warp_str})")
    else:
        print(f"  ✗ {label}  — not found  "
              f"(rejected candidates: {result['rejected']})")


# ── Modes ─────────────────────────────────────────────────────────────────────

def test_image(image_path):
    detector = make_detector()
    frame    = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot open: {image_path}")
        return

    vis, result = analyze_frame(frame, detector,
                                label=os.path.basename(image_path))
    print_result(result)

    out = image_path.rsplit(".", 1)[0] + "_test.jpg"
    cv2.imwrite(out, vis)
    print(f"  Saved: {out}")

    cv2.imshow("ArUco test", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_folder(folder_path):
    detector = make_detector()
    exts     = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files    = []
    for ext in exts:
        for f in glob.glob(os.path.join(folder_path, ext)):
            if "_test" not in f:
                files.append(f)
    files = sorted(files)

    if not files:
        print(f"No images found in {folder_path}")
        return

    print(f"Testing {len(files)} images...\n")

    total_found    = 0
    total_missed   = 0
    warp_scores    = []

    for fpath in files:
        frame        = cv2.imread(fpath)
        vis, result  = analyze_frame(frame, detector,
                                     label=os.path.basename(fpath))
        print_result(result)

        out = fpath.rsplit(".", 1)[0] + "_test.jpg"
        cv2.imwrite(out, vis)

        if result["found"] > 0:
            total_found += 1
            for m in result["markers"]:
                warp_scores.append(m["warp"])
        else:
            total_missed += 1

    # Summary
    print(f"\n{'─'*50}")
    print(f"Results: {total_found}/{len(files)} images detected")
    if warp_scores:
        print(f"Average warp score: {np.mean(warp_scores):.3f}  "
              f"(max: {np.max(warp_scores):.3f})")
        print(f"Warp guide: <0.05 flat  <0.15 slight  <0.30 moderate  >0.30 heavy")


def test_live():
    """Live camera feed — useful for real-time testing of angles and lighting."""
    detector = make_detector()
    cap      = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera. Try --image or --folder instead.")
        return

    print("Live test running.")
    print("  Q     = quit")
    print("  S     = save snapshot")

    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis, result = analyze_frame(frame, detector)

        cv2.imshow("ArUco live test  [Q=quit  S=save]", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%H%M%S")
            path = f"snapshot_{ts}.jpg"
            cv2.imwrite(path, vis)
            snapshot_count += 1
            print(f"Saved snapshot: {path}")
            print_result(result)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{snapshot_count} snapshots saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Test a single image")
    group.add_argument("--folder", help="Test all images in a folder")
    group.add_argument("--live",   action="store_true",
                       help="Live camera feed test")
    args = parser.parse_args()

    if args.image:
        test_image(args.image)
    elif args.folder:
        test_folder(args.folder)
    elif args.live:
        test_live()