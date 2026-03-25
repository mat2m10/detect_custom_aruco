"""
extract_frames.py
-----------------
Extract frames from a video file.

Usage:
    python extract_frames.py --video input.MOV
    python extract_frames.py --video input.MOV --every 10   # 1 frame every 10
    python extract_frames.py --video input.MOV --every 1    # every frame
"""

import cv2
import argparse
import os

def extract(video_path, every_n):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f}fps")
    print(f"Extracting 1 frame every {every_n} frames "
          f"(~{total // every_n} images)")

    out_dir = video_path.rsplit(".", 1)[0] + "_frames"
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i % every_n == 0:
            path = os.path.join(out_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--every", type=int, default=1,
                        help="Extract 1 frame every N (default: 1)")
    args = parser.parse_args()
    extract(args.video, args.every)