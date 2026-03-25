"""
calibrate_camera.py
-------------------
Use your chessboard to calibrate your phone camera.

Usage:
    1. Put all calibration photos in a folder e.g. ./data/calibration/
    2. Run: python calibrate_camera.py --photos ./data/calibration/

Output:
    camera_params.npz  (reuse for all videos with this phone)
"""

import cv2
import numpy as np
import argparse
import glob
import os

CHESSBOARD_SIZE = (7, 7)  # inner corners on a standard 8x8 board
SQUARE_SIZE_MM  = 52.0    # 5.2 cm

# Maximum width to resize images to before detection.
# iPhone photos are 4032px wide — way too large, actually hurts detection.
MAX_WIDTH = 1280


def preprocess(gray):
    """
    Aggressive preprocessing for textured/leather chessboard.
    Order matters: denoise first, then enhance contrast, then sharpen.
    """
    # 1. Bilateral filter — kills texture noise while keeping hard edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. CLAHE — boost local contrast so dark/light squares are more distinct
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 3. Sharpen — recover the actual corner edges after smoothing
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray


def load_and_downscale(fpath):
    """Load image and downscale to MAX_WIDTH, return (img_bgr, scale_factor)."""
    img = cv2.imread(fpath)
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        return img, scale
    return img, 1.0


def calibrate(photo_dir):
    # Build real-world 3D corner coordinates (z=0, flat plane)
    # Points are in mm: (0,0,0), (52,0,0), (104,0,0), ...
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []  # 3D world points
    img_points = []  # 2D image points
    image_size = None

    # Collect image paths (skip _corners previews from previous runs)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in extensions:
        for f in glob.glob(os.path.join(photo_dir, ext)):
            if "_corners" not in f:
                image_files.append(f)
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in {photo_dir}")
        return

    print(f"Found {len(image_files)} images\n")

    successful = 0

    # Detection flags — all three together are much more robust than default
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FILTER_QUADS)

    for fpath in image_files:
        img, scale = load_and_downscale(fpath)
        if img is None:
            print(f"  ! Could not read {os.path.basename(fpath)}, skipping")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_processed = preprocess(gray)

        found, corners = cv2.findChessboardCorners(
            gray_processed, CHESSBOARD_SIZE, flags
        )

        if found:
            # Refine on the processed gray (same image used for detection)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(
                gray_processed, corners, (11, 11), (-1, -1), criteria
            )

            obj_points.append(objp)
            img_points.append(corners)
            image_size = gray_processed.shape[::-1]  # (width, height)
            successful += 1

            # Save preview — draw ON the same downscaled image used for detection
            preview = img.copy()
            cv2.drawChessboardCorners(preview, CHESSBOARD_SIZE, corners, found)
            preview_path = fpath.rsplit(".", 1)[0] + "_corners.jpg"
            cv2.imwrite(preview_path, preview)
            print(f"  ✓ {os.path.basename(fpath)}")
        else:
            print(f"  ✗ {os.path.basename(fpath)}  (corners not found)")

    print(f"\n{successful}/{len(image_files)} images used for calibration")

    if successful == 0:
        print("\nNothing worked. Try:")
        print("  - Better lighting (avoid shadows across the board)")
        print("  - Make sure the full board is visible in each photo")
        print("  - Try from directly overhead first, then add angles")
        return

    if successful < 10:
        print("Warning: fewer than 10 images — calibration may be inaccurate.")
        print("Ideally get 15-20+ from varied angles.\n")

    print("\nComputing calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print(f"Reprojection error: {ret:.4f} px  ", end="")
    if ret < 0.5:
        print("(excellent ✓)")
    elif ret < 1.0:
        print("(good ✓)")
    elif ret < 2.0:
        print("(acceptable — more varied angles would help)")
    else:
        print("(poor — check the _corners.jpg previews, corners may be wrong)")

    np.savez("camera_params.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)

    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    print("\nSaved: camera_params.npz")

    # Per-image reprojection error — helps spot outlier images
    print("\nPer-image reprojection error:")
    errors = []
    for i, (op, ip, rv, tv) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(op, rv, tv, camera_matrix, dist_coeffs)
        err = cv2.norm(ip, projected, cv2.NORM_L2) / len(projected)
        errors.append((err, image_files[i]))
        print(f"  {err:.4f}px  {os.path.basename(image_files[i])}")

    worst = max(errors, key=lambda x: x[0])
    print(f"\nWorst image: {os.path.basename(worst[1])} ({worst[0]:.4f}px)")
    print("If that image has a bad _corners.jpg preview, delete it and rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", required=True,
                        help="Folder containing calibration photos")
    args = parser.parse_args()
    calibrate(args.photos)