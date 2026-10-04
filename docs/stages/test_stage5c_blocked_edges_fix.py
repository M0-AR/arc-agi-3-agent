"""Step 5c: fix the third confirmed issue -- an action that NEVER once
succeeds never gets a delta recorded (match_blob_deltas skips dx=dy=0), so
observe_movement_attempt (gated on `action_id in self.avatar_deltas`) could
never learn anything about it, and the agent retried it forever. Fix:
record the specific (position, action) as a blocked edge directly,
independent of whether a global delta is known for that action -- matching
the standard "remove the observed blocked edge" approach from online grid
pathfinding literature."""
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-md-src/a6a81434-6a0f-4812-b44e-1b9b1d337e66/scratchpad/arc3")
import numpy as np
from step5b_core import Perception, bfs_path

def blank(color=0):
    return np.full((64, 64), color, dtype=np.int64)

# --- Test: an action that NEVER succeeds still gets recorded as a blocked
# edge at the specific position, and BFS routes around it ---
p = Perception()
p.avatar_key = (9, 1)
p.avatar_deltas = {1: (1, 0), 2: (-1, 0)}  # action 3 ("down") never learned -- always blocked here
frame = blank()
frame[10, 10] = 9  # avatar at (10,10)
frame[11, 10] = 7  # a wall sits below it, but we have NO delta for action 3

# simulate 3 failed attempts at action 3 from the same position
blocked_edges = getattr(p, "blocked_edges", None)
assert blocked_edges is not None, "Perception should have a blocked_edges set"
for _ in range(3):
    p.observe_movement_attempt(frame, action_id=3, pos_before=(10, 10), pos_after=(10, 10))

assert ((10, 10), 3) in p.blocked_edges, f"expected (10,10),3 to be a blocked edge, got {p.blocked_edges}"
print("Test 1 (position-specific blocked edge recorded without a known delta) PASSED:", p.blocked_edges)

# --- Test: BFS must not propose a blocked edge as a step, even with a
# "complete" deltas dict that technically allows it geometrically ---
p2 = Perception()
p2.avatar_key = (9, 1)
p2.avatar_deltas = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
p2.wall_colors = set()  # no color-based walls known
p2.blocked_edges = {((10, 10), 3)}  # but we DO know this specific edge is dead
frame2 = blank()
frame2[10, 20] = 6  # target straight down-ish, only reachable via action 3 if not blocked
path = bfs_path((10, 10), [(10, 20)], p2.avatar_deltas, p2.wall_colors, frame2, blocked_edges=p2.blocked_edges)
# with (10,10)->action3 blocked, BFS must NOT start with action 3
assert path is None or path[0] != 3, f"BFS should avoid the known-blocked first edge, got path={path}"
print("Test 2 (BFS respects position-specific blocked edges) PASSED: path =", path)

# --- Test: once a *different* position's same action succeeds, that's a
# SEPARATE edge -- blocking (10,10)->3 must not block (20,20)->3 ---
p3 = Perception()
p3.avatar_key = (9, 1)
p3.avatar_deltas = {3: (0, 1)}
p3.blocked_edges = {((10, 10), 3)}
frame3 = blank()
path3 = bfs_path((20, 20), [(20, 30)], p3.avatar_deltas, p3.wall_colors if hasattr(p3, "wall_colors") else set(), frame3, blocked_edges=p3.blocked_edges)
assert path3 is not None and path3[0] == 3, f"action 3 should still work from a different position, got {path3}"
print("Test 3 (blocking is position-specific, not global) PASSED: path =", path3)

print("\nALL STEP 5c TESTS PASSED")
