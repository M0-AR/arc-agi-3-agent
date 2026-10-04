"""Direct empirical check: when the avatar (color 9) legitimately moved
between consecutive frames, what color were the cells it passed through?
If color 4 ever shows up there, it's proven walkable, not a wall."""
import json
import sys
import numpy as np

AVATAR_COLOR = 9

def load_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)["data"]
            if "frame" not in d or not d["frame"]:
                continue
            frames.append(np.array(d["frame"][0]))
    return frames

def avatar_pos(frame):
    ys, xs = np.where(frame == AVATAR_COLOR)
    if len(xs) == 0:
        return None
    return (int(round(xs.mean())), int(round(ys.mean())))

def analyze(path):
    frames = load_frames(path)
    print(f"\n=== {path.split('/')[-1]} ({len(frames)} frames) ===")
    passed_through_colors = set()
    color4_cells_ever_under_avatar = 0
    moves = 0
    for i in range(len(frames) - 1):
        p_before = avatar_pos(frames[i])
        p_after = avatar_pos(frames[i + 1])
        if p_before is None or p_after is None or p_before == p_after:
            continue
        moves += 1
        x0, y0 = p_before
        x1, y1 = p_after
        # sample the straight line between before/after (inclusive), using the
        # BEFORE frame's colors at those coordinates (what was "under" the path)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            x = round(x0 + (x1 - x0) * s / steps)
            y = round(y0 + (y1 - y0) * s / steps)
            if 0 <= x < 64 and 0 <= y < 64:
                c = int(frames[i][y, x])
                if c != AVATAR_COLOR:
                    passed_through_colors.add(c)
                    if c == 4:
                        color4_cells_ever_under_avatar += 1
    print(f"confirmed real moves: {moves}")
    print(f"colors ever seen along a real movement path (excluding avatar itself): {sorted(passed_through_colors)}")
    print(f"number of path-samples landing on color 4: {color4_cells_ever_under_avatar}")
    if 4 in passed_through_colors:
        print(">>> COLOR 4 WAS WALKED OVER -- it is NOT a wall, our classification was wrong.")
    else:
        print(">>> color 4 never appeared on a confirmed movement path (consistent with, but doesn't prove, wall status)")

for p in sys.argv[1:]:
    analyze(p)
