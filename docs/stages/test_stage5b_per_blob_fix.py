import numpy as np
from step5b_core import Perception, find_blobs, match_blob_deltas, bfs_path, GRID

def blank(color=0):
    return np.full((GRID, GRID), color, dtype=np.int64)

# --- Test 1: find_blobs correctly separates disconnected same-color regions ---
f = blank()
f[10, 10] = 9          # blob A: size 1
f[30:32, 30:32] = 9    # blob B: size 4 (2x2)
mask = np.zeros_like(f, dtype=bool)
blobs = find_blobs(f, mask)
color9 = [b for b in blobs if b["color"] == 9]
assert len(color9) == 2, f"expected 2 separate blobs of color 9, got {len(color9)}"
sizes = sorted(b["size"] for b in color9)
assert sizes == [1, 4], f"expected sizes [1,4], got {sizes}"
print("Test 1 (blob separation) PASSED:", sizes)

# --- Test 2: THE EXACT BUG WE FOUND -- multiple same-color blobs, only one
# moves; per-color centroid would be wrong, per-blob must get it right ---
def make_frame(moving_xy, static_xy_list):
    f = blank()
    mx, my = moving_xy
    f[my, mx] = 9
    for (sx, sy) in static_xy_list:
        f[sy, sx] = 9
    return f

static_blobs = [(50, 50), (5, 55)]  # these never move -- decorations
moving_pos = (10, 10)
prev = make_frame(moving_pos, static_blobs)
p = Perception()
for step in range(4):
    action = 1 if step % 2 == 0 else 2
    moving_pos = (moving_pos[0] + (2 if action == 1 else 0), moving_pos[1] + (0 if action == 1 else 2))
    curr = make_frame(moving_pos, static_blobs)
    p.observe_transition(prev, action, curr)
    prev = curr

assert p.avatar_key is not None, "avatar should have been identified"
assert p.avatar_key[0] == 9, f"avatar color should be 9, got {p.avatar_key}"
assert p.avatar_key[1] == 1, f"avatar should be the size-1 blob (the mover), got size {p.avatar_key[1]}"
pos = p.avatar_position(prev)
assert pos == moving_pos, f"avatar_position should track the MOVING blob, got {pos}, expected {moving_pos}"
print(f"Test 2 (per-blob avatar tracking ignores static same-color decorations) PASSED: avatar_key={p.avatar_key}, pos={pos}")

# --- Test 3: wall learning still works on top of correct blob tracking ---
p3 = Perception()
p3.avatar_key = (9, 1)
p3.avatar_deltas = {1: (1, 0)}
frame = blank()
frame[10, 10] = 9
frame[10, 11] = 7  # wall color 7 directly to the right
p3.observe_movement_attempt(frame, action_id=1, pos_before=(10, 10), pos_after=(10, 10))
assert 7 in p3.wall_colors, f"expected wall color 7, got {p3.wall_colors}"
print("Test 3 (wall learning on correct blob-based position) PASSED:", p3.wall_colors)

# --- Test 4: a legitimately walked-over color must NOT become a wall ---
p4 = Perception()
p4.avatar_key = (9, 1)
p4.avatar_deltas = {1: (1, 0)}
frame = blank(color=4)  # whole floor is color 4
frame[10, 10] = 9
# avatar moves successfully onto a color-4 cell -- pos_after != pos_before
p4.observe_movement_attempt(frame, action_id=1, pos_before=(10, 10), pos_after=(11, 10))
assert 4 not in p4.wall_colors, f"color 4 was walked onto successfully, should never be a wall: {p4.wall_colors}"
print("Test 4 (no false wall on successful move) PASSED")

# --- Test 5: BFS still works end to end with the new Perception class ---
p5 = Perception()
p5.avatar_key = (9, 1)
p5.avatar_deltas = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
p5.wall_colors = {7}
frame = blank()
frame[10, 15] = 7
frame[9, 15] = 7
frame[11, 15] = 7
frame[10, 20] = 6
targets = p5.rare_color_targets(frame)
path = bfs_path((10, 10), targets, p5.avatar_deltas, p5.wall_colors, frame)
assert path is not None
print(f"Test 5 (BFS with new Perception) PASSED: {len(path)}-step path")

print("\nALL STEP 5b TESTS PASSED")
