"""Check whether whole-frame hashing (what our Step 2-4 agent uses) is being
polluted by cells/rows/columns that change constantly regardless of action --
the volatility problem described in the BDR-Pro agent's design."""
import json
import sys
import numpy as np

def analyze(path):
    frames = []
    actions = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)["data"]
            if "frame" not in d or not d["frame"]:
                continue
            frames.append(np.array(d["frame"][0]))
            ai = d.get("action_input", {})
            actions.append(ai.get("id"))

    print(f"\n=== {path.split('/')[-1]} ===")
    print(f"{len(frames)} frames recorded")
    if len(frames) < 2:
        print("too few frames to analyze")
        return

    h, w = frames[0].shape
    change_count = np.zeros((h, w), dtype=int)
    n_transitions = len(frames) - 1

    whole_frame_changed = 0
    for i in range(n_transitions):
        diff = frames[i] != frames[i + 1]
        change_count += diff.astype(int)
        if diff.any():
            whole_frame_changed += 1

    change_freq = change_count / max(n_transitions, 1)
    row_change_freq = (change_count.sum(axis=1) > 0).astype(float)  # simplistic row check
    # proper per-row / per-col change RATE across transitions:
    row_freq = np.zeros(h)
    col_freq = np.zeros(w)
    for i in range(n_transitions):
        diff = frames[i] != frames[i + 1]
        row_freq += diff.any(axis=1).astype(float)
        col_freq += diff.any(axis=0).astype(float)
    row_freq /= max(n_transitions, 1)
    col_freq /= max(n_transitions, 1)

    volatile_cells = (change_freq >= 0.20).sum()
    volatile_rows = (row_freq >= 0.40).sum()
    volatile_cols = (col_freq >= 0.40).sum()
    total_cells = h * w

    print(f"whole-frame hash changed on {whole_frame_changed}/{n_transitions} transitions "
          f"({100*whole_frame_changed/max(n_transitions,1):.0f}%)")
    print(f"cells changing in >=20% of transitions (volatility-mask threshold): "
          f"{volatile_cells}/{total_cells} ({100*volatile_cells/total_cells:.1f}%)")
    print(f"rows changing in >=40% of transitions: {volatile_rows}/{h}")
    print(f"cols changing in >=40% of transitions: {volatile_cols}/{w}")

    if volatile_cells > 0:
        ys, xs = np.where(change_freq >= 0.20)
        print(f"example volatile cell coords (up to 10): {list(zip(xs[:10].tolist(), ys[:10].tolist()))}")
        # what colors appear at the most volatile cell over time?
        top_idx = np.argmax(change_freq)
        ty, tx = np.unravel_index(top_idx, change_freq.shape)
        vals = [int(f[ty, tx]) for f in frames]
        print(f"most volatile cell (x={tx},y={ty}) value sequence across frames: {vals}")
    else:
        print("No volatile cells found above threshold -- whole-frame hashing looks SAFE for this run.")

    # How many *distinct* whole-frame hashes vs total frames (novelty check)
    import hashlib
    hashes = [hashlib.md5(str(f.tolist()).encode()).hexdigest() for f in frames]
    print(f"distinct whole-frame states: {len(set(hashes))}/{len(frames)}")


for path in sys.argv[1:]:
    analyze(path)
