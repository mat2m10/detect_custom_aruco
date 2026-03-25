"""
interpolate_pose.py
-------------------
Takes the pose_data.csv from extract_pose.py, filters spikes,
and linearly interpolates missing frames.

Usage:
    python interpolate_pose.py --csv data/mov/test_frames/pose_data.csv
    python interpolate_pose.py --csv data/mov/test_frames/pose_data.csv --plot
"""

import numpy as np
import csv
import argparse
import os

# Max degrees a rotation axis can jump between consecutive detected frames
# before being considered a spike (false positive detection)
SPIKE_THRESHOLD_DEG = 30.0


def load_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame":    int(row["frame"]),
                "file":     row["file"],
                "detected": int(row["detected"]),
                "id":       row["id"],
                "size_px":  float(row["size_px"]) if row["size_px"] else None,
                "rx":       float(row["rx"]) if row["rx"] else None,
                "ry":       float(row["ry"]) if row["ry"] else None,
                "rz":       float(row["rz"]) if row["rz"] else None,
                "tx":       float(row["tx"]) if row["tx"] else None,
                "ty":       float(row["ty"]) if row["ty"] else None,
                "tz":       float(row["tz"]) if row["tz"] else None,
            })
    return rows


def filter_spikes(rows):
    """
    Mark detections as invalid if any rotation axis jumps more than
    SPIKE_THRESHOLD_DEG from the previous valid detection.
    These are almost certainly false positives.
    """
    last_valid = None
    spike_count = 0

    for row in rows:
        if row["detected"] != 1 or row["rx"] is None:
            continue

        if last_valid is not None:
            jump_rx = abs(row["rx"] - last_valid["rx"])
            jump_ry = abs(row["ry"] - last_valid["ry"])
            jump_rz = abs(row["rz"] - last_valid["rz"])

            # Handle angle wraparound (e.g. 179° → -179° is only 2° jump)
            jump_rx = min(jump_rx, 360 - jump_rx)
            jump_ry = min(jump_ry, 360 - jump_ry)
            jump_rz = min(jump_rz, 360 - jump_rz)

            if (jump_rx > SPIKE_THRESHOLD_DEG or
                jump_ry > SPIKE_THRESHOLD_DEG or
                jump_rz > SPIKE_THRESHOLD_DEG):
                # Mark as spike — treat as missing
                row["detected"] = 0
                row["rx"] = row["ry"] = row["rz"] = None
                row["tx"] = row["ty"] = row["tz"] = None
                row["spike"] = True
                spike_count += 1
                continue

        row["spike"] = False
        last_valid = row

    return rows, spike_count


def interpolate(rows):
    """
    Linearly interpolate rx, ry, rz, tx, ty, tz for missing frames
    between valid detections.
    """
    axes = ["rx", "ry", "rz", "tx", "ty", "tz"]

    # Find indices of valid detections
    valid_idx = [i for i, r in enumerate(rows) if r["detected"] == 1]

    if len(valid_idx) < 2:
        print("Not enough valid detections to interpolate.")
        return rows, 0

    interpolated_count = 0

    for i in range(len(valid_idx) - 1):
        i0 = valid_idx[i]
        i1 = valid_idx[i + 1]

        if i1 - i0 == 1:
            continue  # consecutive frames, nothing to fill

        r0 = rows[i0]
        r1 = rows[i1]

        # Only interpolate if gap isn't too large (>60 frames = 2sec at 30fps)
        # Beyond that, hold last known value instead
        gap = i1 - i0
        if gap > 60:
            # Hold last known value
            for j in range(i0 + 1, i1):
                rows[j]["interp"] = "held"
                for ax in axes:
                    rows[j][ax] = r0[ax]
                interpolated_count += 1
            continue

        # Linear interpolation
        for j in range(i0 + 1, i1):
            t = (j - i0) / (i1 - i0)  # 0.0 → 1.0
            rows[j]["interp"] = "linear"
            for ax in axes:
                if r0[ax] is not None and r1[ax] is not None:
                    rows[j][ax] = r0[ax] + t * (r1[ax] - r0[ax])
            interpolated_count += 1

    # Fill frames before first detection and after last detection
    # with the nearest known value
    if valid_idx:
        first_val = rows[valid_idx[0]]
        for i in range(0, valid_idx[0]):
            rows[i]["interp"] = "held"
            for ax in axes:
                rows[i][ax] = first_val[ax]

        last_val = rows[valid_idx[-1]]
        for i in range(valid_idx[-1] + 1, len(rows)):
            rows[i]["interp"] = "held"
            for ax in axes:
                rows[i][ax] = last_val[ax]

    return rows, interpolated_count


def save_csv(rows, out_path):
    fieldnames = ["frame", "file", "detected", "interp", "spike",
                  "id", "size_px", "rx", "ry", "rz", "tx", "ty", "tz"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Round floats for readability
            out = dict(row)
            for ax in ["rx", "ry", "rz", "tx", "ty", "tz"]:
                if out[ax] is not None:
                    out[ax] = round(out[ax], 3)
            out.setdefault("interp", "detected")
            out.setdefault("spike",  False)
            writer.writerow(out)


def process(csv_path, do_plot):
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} frames from {csv_path}")

    detected_before = sum(1 for r in rows if r["detected"] == 1)
    print(f"Detected before filtering: {detected_before}/{len(rows)}")

    # Step 1: filter spikes
    rows, spike_count = filter_spikes(rows)
    detected_after = sum(1 for r in rows if r["detected"] == 1)
    print(f"Spikes removed:            {spike_count}")
    print(f"Detected after filtering:  {detected_after}/{len(rows)}")

    # Step 2: interpolate
    rows, interp_count = interpolate(rows)
    print(f"Frames interpolated:       {interp_count}")

    total_valid = sum(1 for r in rows if r["rx"] is not None)
    print(f"Total frames with Rx:      {total_valid}/{len(rows)}")

    # Save
    out_path = csv_path.replace(".csv", "_interpolated.csv")
    save_csv(rows, out_path)
    print(f"\nSaved: {out_path}")

    if do_plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            frames = [r["frame"] for r in rows]
            rx_all = [r["rx"]    for r in rows]
            ry_all = [r["ry"]    for r in rows]
            rz_all = [r["rz"]    for r in rows]

            # Separate by source for coloring
            det_f  = [r["frame"] for r in rows if r.get("interp","detected") == "detected" and r["rx"] is not None]
            det_rx = [r["rx"]    for r in rows if r.get("interp","detected") == "detected" and r["rx"] is not None]

            int_f  = [r["frame"] for r in rows if r.get("interp") == "linear"]
            int_rx = [r["rx"]    for r in rows if r.get("interp") == "linear"]

            fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
            fig.suptitle("Pose data — after spike filter + interpolation")

            for ax_plot, vals, color, label in [
                (axes[0], (rx_all, det_f, det_rx, int_f, int_rx), "red",   "Rx (tilt) °"),
                (axes[1], (ry_all, [], [], [], []),                "green", "Ry (top/bottom) °"),
                (axes[2], (rz_all, [], [], [], []),                "blue",  "Rz (turntable) °"),
            ]:
                ax_plot.plot(frames, vals[0], color=color, alpha=0.3, linewidth=1)
                ax_plot.axhline(0, color="gray", linestyle="--", linewidth=0.8)
                ax_plot.set_ylabel(label)

            # Highlight detected vs interpolated on Rx
            axes[0].scatter(det_f, det_rx, color="red",    s=4, zorder=3, label="detected")
            axes[0].scatter(int_f, int_rx, color="orange", s=4, zorder=3, label="interpolated")
            axes[0].legend(loc="upper right", fontsize=8)

            axes[2].set_xlabel("Frame number")
            plt.tight_layout()

            plot_path = csv_path.replace(".csv", "_interpolated_plot.png")
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved: {plot_path}")
            plt.show()

        except ImportError:
            print("matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True, help="pose_data.csv from extract_pose.py")
    parser.add_argument("--plot", action="store_true", help="Plot the result")
    args = parser.parse_args()
    process(args.csv, args.plot)