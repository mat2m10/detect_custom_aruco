# Camera Calibration

Before we can estimate the 3D pose of an ArUco marker (position + orientation in space), we need to know how our specific camera and lens distort reality. This is called **camera calibration**.

## What it does

A camera is not a perfect mathematical object. Two things need to be measured:

1. **Lens distortion** — the lens bends light, so straight lines in the real world appear slightly curved in the photo (barrel or pincushion distortion).
2. **Intrinsic matrix** — the camera's focal length and optical center in pixels, captured in a 3×3 matrix:

```
[ fx   0   cx ]
[  0  fy   cy ]
[  0   0    1 ]
```

Where `fx, fy` are the focal lengths and `cx, cy` is the optical center (not always exactly the middle of the sensor).

To measure both, we show the camera a pattern whose exact geometry we know — a checkerboard. Because every square is the same size and every corner is at a known position, OpenCV can work backwards and figure out exactly how the lens is distorting reality.

## How to calibrate

### 1. What you need

- A checkerboard or chessboard (a real chess board works perfectly)
- Note the **square size in mm** and the number of **inner corners** (a standard 8×8 chess board has 7×7 inner corners)
- Change those values in calibrate_camera.py

### 2. Take 25–35 photos

- Keep the board flat and stationary, move the camera around it
- Cover a wide variety of angles: tilt toward you, away, left, right, from corners
- Vary the distance: some close, some far
- Keep the full board visible in every shot
- Make sure photos are in focus

The more varied the angles, the better the calibration.

### 3. Run the script

```bash
python calibrate_camera.py --photos ./data/calibration/
```

The script will:
- Try to detect the 7×7 grid of inner corners in each photo
- Save a `_corners.jpg` preview next to each photo — check these to make sure the colored dots sit on actual square intersections
- Compute the calibration and report a **reprojection error** in pixels
- Save the result to `camera_params.npz`

### 4. Check the result

A good `_corners.jpg` preview looks like this: colored × marks sitting precisely on corner intersections, with rows connecting them in order (red = row 1, orange = row 2, etc.).

Reprojection error guide:
- Under 0.5px → excellent
- 0.5–1.0px → good
- 1.0–2.0px → acceptable
- Over 2.0px → poor, take more photos from more varied angles

### 5. Save `camera_params.npz`

This file is reusable — as long as you use the same phone with the same camera settings, you never need to recalibrate. Keep it in your project root.

## Notes

- **Texture matters**: a heavily textured board (e.g. leather) is harder to detect than a printed paper checkerboard. The script includes bilateral filtering + CLAHE preprocessing to handle this.
- **iPhone HEIC files**: convert to JPG first, or use `pillow-heif` to handle them in Python.
- **Image size**: the script downscales to 1280px before detection — large phone images (4032px) actually hurt corner detection accuracy.
