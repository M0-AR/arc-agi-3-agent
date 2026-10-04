"""Step 5d: rare_color_targets must return one target per BLOB (centroid),
not one per cell -- fixes a confirmed bug where a single large rare-colored
object (e.g. 84 cells, just under the 2%-of-grid threshold) was treated as
84 separate targets, so incidental wandering over a few of its cells could
exhaust the whole object as "visited" without ever reaching its center."""
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-md-src/a6a81434-6a0f-4812-b44e-1b9b1d337e66/scratchpad/arc3")
import numpy as np
from step5b_core import Perception

def blank(color=0):
    return np.full((64, 64), color, dtype=np.int64)

# --- Test 1: one blob of a rare color -> exactly ONE target (its centroid),
# not N targets for its N cells ---
p = Perception()
frame = blank()
frame[20:25, 20:25] = 6  # a 5x5 rare-colored blob, 25 cells (~0.6% of grid)
targets = p.rare_color_targets(frame)
assert len(targets) == 1, f"expected exactly 1 target for one blob, got {len(targets)}: {targets}"
assert targets[0] == (22, 22), f"expected centroid (22,22), got {targets[0]}"
print("Test 1 (one blob = one target, at its centroid) PASSED:", targets)

# --- Test 2: two SEPARATE blobs of the same rare color -> two targets ---
p2 = Perception()
frame2 = blank()
frame2[5:8, 5:8] = 6     # blob A, centroid (6,6)
frame2[50:53, 50:53] = 6  # blob B, centroid (51,51)
targets2 = p2.rare_color_targets(frame2)
assert len(targets2) == 2, f"expected 2 separate targets, got {targets2}"
assert set(targets2) == {(6, 6), (51, 51)}, f"wrong centroids: {targets2}"
print("Test 2 (two disconnected blobs = two separate targets) PASSED:", targets2)

# --- Test 3: visiting ONE blob's centroid must not exhaust a DIFFERENT
# blob's centroid -- this is the actual exhaustion-prevention property ---
visited = {(6, 6)}  # simulating visited_positions after reaching blob A
remaining = [t for t in targets2 if t not in visited]
assert remaining == [(51, 51)], f"blob B should still be targetable: {remaining}"
print("Test 3 (visiting one blob leaves other distinct blobs targetable) PASSED")

# --- Test 4: a large blob just under the rarity threshold is still ONE
# target, not proportionally many (reproduces the real observed case:
# ~84 cells out of 4096 = ~2.05%, right at the edge) ---
p4 = Perception()
frame4 = blank()
# an 9x9 blob = 81 cells (~1.98%, just under the 2% cutoff)
frame4[10:19, 10:19] = 11
targets4 = p4.rare_color_targets(frame4)
assert len(targets4) == 1, f"a single large blob must still be ONE target, got {len(targets4)}"
print("Test 4 (large near-threshold blob is still a single target) PASSED:", targets4)

print("\nALL STEP 5d TESTS PASSED")
