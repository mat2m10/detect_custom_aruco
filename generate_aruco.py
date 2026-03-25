"""
generate_aruco.py
-----------------
Generate an ArUco marker image and print an ASCII version for hand-drawing.

Usage:
    python generate_aruco.py --id 0
    python generate_aruco.py --id 0 --size 400 --output marker_0.png

The ASCII output in the terminal is what you copy onto paper with a grid.
█ = fill black,  · = leave white
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

DICTIONARY = aruco.DICT_4X4_50  # 4x4 grid, 50 possible IDs (we only need 1)
BORDER_BITS = 1                  # the black border around the data cells


def generate(marker_id, size_px, output_path):
    dictionary = aruco.getPredefinedDictionary(DICTIONARY)

    # Generate marker image (size_px × size_px)
    marker_img = np.zeros((size_px, size_px), dtype=np.uint8)
    aruco.generateImageMarker(dictionary, marker_id, size_px, marker_img, BORDER_BITS)

    # Save image
    cv2.imwrite(output_path, marker_img)
    print(f"Saved: {output_path}")

    # --- ASCII representation for hand-drawing ---
    # The marker is a 6×6 grid (4 data cells + 1 border cell on each side)
    total_cells = 4 + 2 * BORDER_BITS  # = 6
    cell_size = size_px // total_cells

    print(f"\nMarker ID {marker_id} — 4×4 dictionary")
    print(f"Grid: {total_cells}×{total_cells} cells (including border)\n")

    # Sample the center of each cell
    print("  +" + "---+" * total_cells)
    for row in range(total_cells):
        line = "  |"
        for col in range(total_cells):
            cy = int((row + 0.5) * cell_size)
            cx = int((col + 0.5) * cell_size)
            # Clamp to image bounds
            cy = min(cy, size_px - 1)
            cx = min(cx, size_px - 1)
            pixel = marker_img[cy, cx]
            cell = " █ " if pixel < 128 else " · "
            line += cell + "|"
        print(line)
        print("  +" + "---+" * total_cells)

    print()
    print("Legend:  █ = black cell    · = white cell")
    print()
    print("How to draw:")
    print("  1. Draw a 6×6 grid on paper (use a ruler!)")
    print("  2. Fill in every cell marked █ with black ink")
    print("  3. Leave every cell marked · white")
    print("  4. The outer ring is always all-black (the border)")
    print()
    print("Tips:")
    print("  - Bigger = easier to detect. At least 3×3 cm total recommended.")
    print("  - Clean straight lines matter more than perfect square sizes.")
    print("  - Use matte paper to avoid glare under painting lights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",     type=int, default=0,
                        help="Marker ID to generate (0–49)")
    parser.add_argument("--size",   type=int, default=400,
                        help="Output image size in pixels (default: 400)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: marker_<id>.png)")
    args = parser.parse_args()

    if not 0 <= args.id <= 49:
        print("Error: ID must be between 0 and 49 for DICT_4X4_50")
        exit(1)

    output = args.output or f"marker_{args.id}.png"
    generate(args.id, args.size, output)