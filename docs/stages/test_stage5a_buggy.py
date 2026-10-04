import numpy as np
from step5_core import Perception, bfs_path, GRID

def blank(color=0):
    return np.full((GRID, GRID), color, dtype=np.int64)

# --- Test 1: volatility masking detects a flickering cell ---
p = Perception()
frame_a = blank()
frame_a[40, 39] = 12
frame_b = blank()
frame_b[40, 39] = 3
# alternate the flicker cell for 12 transitions, nothing else ever changes
frames = [frame_a if i % 2 == 0 else frame_b for i in range(13)]
for i in range(len(frames) - 1):
    p.update_volatility(frames[i], frames[i + 1])
assert p.volatile_mask[40, 39] == True, "flickering cell should be masked"
assert p.volatile_mask.sum() == 1, f"only the flickering cell should be masked, got {p.volatile_mask.sum()}"
h1 = p.masked_hash(frame_a)
h2 = p.masked_hash(frame_b)
assert h1 == h2, "masked hash should be IDENTICAL despite the flicker -- this is the whole point"
print("Test 1 (volatility masking) PASSED")

# --- Test 2: avatar detection from consistent per-action color movement ---
p2 = Perception()
# color 5 = avatar. action 1 -> moves it (dx=+1,dy=0). action 2 -> moves it (dx=0,dy=+1).
# color 7 = autonomous decoration that ALWAYS drifts (dx=0,dy=-1) regardless of action.
def make_frame(avatar_xy, deco_xy):
    f = blank()
    ax, ay = avatar_xy
    dx, dy = deco_xy
    f[ay, ax] = 5
    f[dy, dx] = 7
    return f

avatar_pos = (10, 10)
deco_pos = (30, 30)
f_prev = make_frame(avatar_pos, deco_pos)
for step in range(6):
    action = 1 if step % 2 == 0 else 2
    if action == 1:
        avatar_pos = (avatar_pos[0] + 1, avatar_pos[1])
    else:
        avatar_pos = (avatar_pos[0], avatar_pos[1] + 1)
    deco_pos = (deco_pos[0], deco_pos[1] - 1)  # autonomous, same regardless of action
    f_curr = make_frame(avatar_pos, deco_pos)
    p2.observe_transition(f_prev, action, f_curr)
    f_prev = f_curr

assert p2.avatar_color == 5, f"expected avatar_color=5, got {p2.avatar_color}"
assert p2.avatar_deltas.get(1) == (1, 0), f"action1 delta wrong: {p2.avatar_deltas}"
assert p2.avatar_deltas.get(2) == (0, 1), f"action2 delta wrong: {p2.avatar_deltas}"
print("Test 2 (avatar detection, excludes autonomous decoration) PASSED:", p2.avatar_color, p2.avatar_deltas)

# --- Test 3: wall detection when a known-moving action fails to move the avatar ---
p3 = Perception()
p3.avatar_color = 5
p3.avatar_deltas = {1: (1, 0)}
frame = blank()
frame[10, 11] = 9  # wall color 9 sits directly to the right of avatar at (10,10)
pos_before = (10, 10)
pos_after = (10, 10)  # didn't move -- blocked
p3.observe_movement_attempt(frame, action_id=1, pos_before=pos_before, pos_after=pos_after)
assert 9 in p3.wall_colors, f"expected color 9 learned as wall, got {p3.wall_colors}"
print("Test 3 (wall learning on blocked movement) PASSED:", p3.wall_colors)

# --- Test 4: BFS pathfinding around a wall to reach a rare-colored target ---
p4 = Perception()
p4.avatar_color = 5
p4.avatar_deltas = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}  # right, left, down, up
p4.wall_colors = {9}
frame = blank()
# a wall strip blocking the direct path right, with a gap to go around via down
frame[10, 15] = 9
frame[11, 15] = 9
frame[9, 15] = 9
frame[10, 20] = 6  # rare target color far to the right
targets = p4.rare_color_targets(frame)
assert (20, 10) in targets
path = bfs_path(start=(10, 10), targets=targets, deltas=p4.avatar_deltas, wall_colors=p4.wall_colors, frame=frame)
assert path is not None, "should find a path around the wall"
# simulate following the path and check we actually reach (20,10) without crossing a wall cell
x, y = 10, 10
for action_id in path:
    dx, dy = p4.avatar_deltas[action_id]
    x, y = x + dx, y + dy
    assert frame[y, x] != 9, f"path crosses a wall at ({x},{y})"
assert (x, y) == (20, 10), f"path should end at target, ended at ({x},{y})"
print(f"Test 4 (BFS pathfinding around wall) PASSED: {len(path)}-step path, ended at ({x},{y})")

print("\nALL STEP 5 TESTS PASSED")
