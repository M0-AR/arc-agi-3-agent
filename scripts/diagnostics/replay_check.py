"""Replay a real recorded game session through the new per-blob logic
(offline, no live API calls, and no dependency on the unreliable recorded
action_id -- see earlier finding that it's always 0). This verifies the
actual fix (blob matching by color+size) directly against real frames:
for each distinct (color, size) blob, does it trace a plausible, mostly-
contiguous trajectory (a real moving object), or does it stay frozen (a
static decoration)?
"""
import json
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-md-src/a6a81434-6a0f-4812-b44e-1b9b1d337e66/scratchpad/arc3")
from collections import defaultdict
import numpy as np
from step5b_core import find_blobs

def load(path):
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)["data"]
            if "frame" not in d or not d["frame"]:
                continue
            frames.append(np.array(d["frame"][0]))
    return frames

def replay(path):
    frames = load(path)
    empty_mask = np.zeros((64, 64), dtype=bool)
    # track positions of each (color,size) key across all frames
    trajectories = defaultdict(list)
    for f in frames:
        blobs = find_blobs(f, empty_mask)
        for b in blobs:
            key = (b["color"], b["size"])
            trajectories[key].append((round(b["cx"]), round(b["cy"])))

    print(f"\n=== {path.split('/')[-1]} ({len(frames)} frames) ===")
    # focus on color 9 (what we previously, wrongly, called "the avatar")
    color9_keys = [k for k in trajectories if k[0] == 9]
    for key in sorted(color9_keys, key=lambda k: k[1]):
        positions = trajectories[key]
        n_distinct = len(set(positions))
        total_movement = sum(
            abs(positions[i][0] - positions[i-1][0]) + abs(positions[i][1] - positions[i-1][1])
            for i in range(1, len(positions))
        )
        label = "MOVER" if n_distinct > 1 else "static"
        print(f"  color=9 size={key[1]:3d}: seen in {len(positions)} frames, "
              f"{n_distinct} distinct positions, total movement={total_movement} -> {label}")

for p in sys.argv[1:]:
    replay(p)
