"""Unit tests for the agent's perception/navigation logic, against synthetic
frames. Each test corresponds to a specific, evidenced bug found during
development against the REAL ARC-AGI-3 service -- see ../docs/FINDINGS.md
for the live debugging story behind each one. Run with:

    python -m pytest tests/test_perception.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

import numpy as np

# my_agent.py imports `arcengine` and `agents.agent`, which only exist
# inside the ARC-AGI-3-Agents runtime (see README for setup). The pieces
# under test here -- Perception, bfs_path, find_blobs, match_blob_deltas --
# have no dependency on either, so we import the module with those two
# lines stubbed out rather than requiring the full runtime just to test
# pure logic.
import types
_src = open(os.path.join(os.path.dirname(__file__), "..", "agent", "my_agent.py")).read()
_src = _src.replace("from arcengine import FrameData, GameAction, GameState", "")
_src = _src.replace("from agents.agent import Agent", "")
_mod = types.ModuleType("my_agent")
# Lightweight stand-ins so MyAgent's class body (method signatures reference
# these as type annotations) can be exec'd without the real ARC-AGI-3
# runtime installed. Only Perception/bfs_path/find_blobs/match_blob_deltas
# are actually exercised by these tests -- MyAgent itself needs the real
# runtime (see README) and is not unit-tested here.
_mod.__dict__["FrameData"] = object
_mod.__dict__["Agent"] = object
_mod.__dict__["GameAction"] = type("GameAction", (), {"RESET": type("R", (), {"value": 0})()})
_mod.__dict__["GameState"] = type("GameState", (), {"NOT_PLAYED": 0, "GAME_OVER": 1, "WIN": 2})
exec(compile(_src, "my_agent.py", "exec"), _mod.__dict__)

Perception = _mod.Perception
bfs_path = _mod.bfs_path
find_blobs = _mod.find_blobs
match_blob_deltas = _mod.match_blob_deltas


def blank(color=0):
    return np.full((64, 64), color, dtype=np.int64)


# ============================================================
# Bug #1: whole-frame hashing polluted by background animation
# ============================================================

def test_volatility_masking_ignores_flicker():
    p = Perception()
    frame_a, frame_b = blank(), blank()
    frame_a[40, 39] = 12
    frame_b[40, 39] = 3
    frames = [frame_a if i % 2 == 0 else frame_b for i in range(13)]
    for i in range(len(frames) - 1):
        p.update_volatility(frames[i], frames[i + 1])
    assert p.volatile_mask[40, 39]
    assert p.volatile_mask.sum() == 1
    assert p.masked_hash(frame_a) == p.masked_hash(frame_b)


# ============================================================
# Bug #2: per-color centroid averaging mixed a moving object
# with static same-colored decorations (measured on real ls20
# frames: color 9 was 5 separate blobs, only 1 ever moved)
# ============================================================

def test_blob_separation():
    f = blank()
    f[10, 10] = 9
    f[30:32, 30:32] = 9
    blobs = find_blobs(f, np.zeros_like(f, dtype=bool))
    color9 = [b for b in blobs if b["color"] == 9]
    assert sorted(b["size"] for b in color9) == [1, 4]


def test_per_blob_avatar_tracking_ignores_static_decorations():
    def make_frame(moving_xy, static_xy_list):
        f = blank()
        mx, my = moving_xy
        f[my, mx] = 9
        for sx, sy in static_xy_list:
            f[sy, sx] = 9
        return f

    static_blobs = [(50, 50), (5, 55)]
    moving_pos = (10, 10)
    prev = make_frame(moving_pos, static_blobs)
    p = Perception()
    for step in range(4):
        action = 1 if step % 2 == 0 else 2
        moving_pos = (moving_pos[0] + (2 if action == 1 else 0),
                      moving_pos[1] + (0 if action == 1 else 2))
        curr = make_frame(moving_pos, static_blobs)
        p.observe_transition(prev, action, curr,
                              find_blobs(prev, p.volatile_mask), find_blobs(curr, p.volatile_mask))
        prev = curr
    assert p.avatar_key == (9, 1)  # color 9, the size-1 (moving) blob
    assert p.avatar_position(find_blobs(prev, p.volatile_mask)) == moving_pos


# ============================================================
# Bug #3: an action that never once succeeds never gets a delta
# recorded, so wall/block detection (gated on a known delta)
# could never learn anything about it -> permanent freeze
# ============================================================

def test_blocked_edge_recorded_without_a_known_delta():
    p = Perception()
    p.avatar_key = (9, 1)
    p.avatar_deltas = {1: (1, 0), 2: (-1, 0)}  # no delta for action 3
    frame = blank()
    frame[10, 10] = 9
    for _ in range(3):
        p.observe_movement_attempt(frame, action_id=3, pos_before=(10, 10), pos_after=(10, 10))
    assert ((10, 10), 3) in p.blocked_edges


def test_bfs_respects_position_specific_blocked_edges():
    p = Perception()
    p.avatar_key = (9, 1)
    p.avatar_deltas = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
    p.blocked_edges = {((10, 10), 3)}
    frame = blank()
    frame[20, 10] = 6
    path = bfs_path((10, 10), [(10, 20)], p.avatar_deltas, p.wall_colors, frame,
                     blocked_edges=p.blocked_edges)
    assert path is None or path[0] != 3


def test_blocking_is_position_specific_not_global():
    p = Perception()
    p.avatar_deltas = {3: (0, 1)}
    p.blocked_edges = {((10, 10), 3)}
    frame = blank()
    path = bfs_path((20, 20), [(20, 30)], p.avatar_deltas, set(), frame,
                     blocked_edges=p.blocked_edges)
    assert path is not None and path[0] == 3  # still works from a DIFFERENT position


def test_no_false_wall_on_successful_move():
    p = Perception()
    p.avatar_key = (9, 1)
    p.avatar_deltas = {1: (1, 0)}
    frame = blank(color=4)
    frame[10, 10] = 9
    p.observe_movement_attempt(frame, action_id=1, pos_before=(10, 10), pos_after=(11, 10))
    assert 4 not in p.wall_colors  # walked onto it successfully -- never a wall


# ============================================================
# Bug #4: per-cell target lists let incidental wandering over a
# few cells of one large object "exhaust" it entirely
# ============================================================

def test_one_blob_is_one_target_at_its_centroid():
    p = Perception()
    frame = blank()
    frame[20:25, 20:25] = 6  # 5x5 blob, 25 cells
    targets = p.rare_color_targets(frame)
    assert targets == [(22, 22)]


def test_two_separate_blobs_are_two_targets():
    p = Perception()
    frame = blank()
    frame[5:8, 5:8] = 6
    frame[50:53, 50:53] = 6
    targets = p.rare_color_targets(frame)
    assert set(targets) == {(6, 6), (51, 51)}


def test_visiting_one_blob_leaves_other_blobs_targetable():
    p = Perception()
    frame = blank()
    frame[5:8, 5:8] = 6
    frame[50:53, 50:53] = 6
    visited = {(6, 6)}
    remaining = [t for t in p.rare_color_targets(frame) if t not in visited]
    assert remaining == [(51, 51)]


def test_bfs_end_to_end_with_wall_avoidance():
    p = Perception()
    p.avatar_key = (9, 1)
    p.avatar_deltas = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
    p.wall_colors = {7}
    frame = blank()
    frame[9, 15] = frame[10, 15] = frame[11, 15] = 7
    frame[10, 20] = 6
    targets = p.rare_color_targets(frame)
    path = bfs_path((10, 10), targets, p.avatar_deltas, p.wall_colors, frame)
    assert path is not None
    x, y = 10, 10
    for a in path:
        dx, dy = p.avatar_deltas[a]
        x, y = x + dx, y + dy
        assert frame[y, x] != 7
    assert (x, y) == (20, 10)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
