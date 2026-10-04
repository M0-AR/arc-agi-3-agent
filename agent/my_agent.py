"""ARC-AGI-3 agent -- object-aware navigation built incrementally in five
verified stages. See ../docs/FINDINGS.md for the full story, including four
real bugs found through direct evidence (not guessed) and fixed one at a
time, each with its own regression test.

Stage summary:
  1. Respect `available_actions` instead of guessing blindly.
  2. Track which actions actually change the frame (change-rate).
  3. Build a real state graph from play; prefer unexplored edges (BFS
     frontier).
  4. Learn which (state, action) pairs advanced a level (Monte-Carlo credit
     assignment), reused across in-episode resets.
  5. Object-aware navigation: volatility-masked hashing, per-blob avatar
     tracking, position-specific blocked-edge learning, and blob-centroid
     BFS targeting -- replacing Stage 1-4's blind exploration once the
     avatar is identified, falling back to it otherwise.

Set ARC3_AGENT_DEBUG=1 to print per-turn perception state to stderr.
"""
import hashlib
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

GRID = 64
REGION_SIZE = 8
MIN_TRANSITIONS_FOR_MASK = 10
VOLATILE_CELL_THRESH = 0.20
RARE_COLOR_MAX_FRAC = 0.02
DEBUG = os.environ.get("ARC3_AGENT_DEBUG") == "1"


def find_blobs(frame: np.ndarray, ignore_mask: np.ndarray) -> List[dict]:
    """4-connected flood fill. One dict per blob: {color, size, cx, cy}."""
    h, w = frame.shape
    visited = np.zeros((h, w), dtype=bool)
    blobs = []
    for y in range(h):
        for x in range(w):
            if visited[y, x] or ignore_mask[y, x]:
                continue
            color = int(frame[y, x])
            stack = [(y, x)]
            visited[y, x] = True
            xs, ys = [], []
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w and not visited[ny, nx]
                            and not ignore_mask[ny, nx] and frame[ny, nx] == color):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            blobs.append({"color": color, "size": len(xs),
                           "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys)})
    return blobs


def match_blob_deltas(prev_blobs: List[dict], curr_blobs: List[dict]) -> List[dict]:
    """Match same-color, same-size blobs between frames by nearest centroid.
    Requiring equal size is what prevents a moving object from being
    confused with a differently-sized static decoration of the same color
    (Bug #2 -- see docs/FINDINGS.md)."""
    prev_by_color: Dict[int, List[dict]] = defaultdict(list)
    curr_by_color: Dict[int, List[dict]] = defaultdict(list)
    for b in prev_blobs:
        prev_by_color[b["color"]].append(b)
    for b in curr_blobs:
        curr_by_color[b["color"]].append(b)
    results = []
    for color, p_list in prev_by_color.items():
        c_list = curr_by_color.get(color, [])
        used: Set[int] = set()
        for p in p_list:
            best_i, best_dist = None, None
            for i, c in enumerate(c_list):
                if i in used or c["size"] != p["size"]:
                    continue
                dist = abs(c["cx"] - p["cx"]) + abs(c["cy"] - p["cy"])
                if best_dist is None or dist < best_dist:
                    best_dist, best_i = dist, i
            if best_i is not None:
                used.add(best_i)
                c = c_list[best_i]
                results.append({"color": color, "size": p["size"],
                                 "dx": round(c["cx"] - p["cx"]), "dy": round(c["cy"] - p["cy"])})
    return results


class Perception:
    """Volatility-masked state hashing + per-blob avatar/wall/target
    detection. See docs/FINDINGS.md for why each piece exists -- every one
    was added to fix a specific, evidenced failure, not speculatively."""

    def __init__(self) -> None:
        self.change_count = np.zeros((GRID, GRID), dtype=np.int64)
        self.n_transitions = 0
        self.volatile_mask = np.zeros((GRID, GRID), dtype=bool)

        self.blob_deltas_by_action: Dict[int, Dict[Tuple[int, int], Counter]] = {}
        self.avatar_key: Optional[Tuple[int, int]] = None  # (color, size)
        self.avatar_deltas: Dict[int, Tuple[int, int]] = {}
        self.wall_colors: Set[int] = set()
        # (position, action_id) pairs observed to produce zero movement from
        # that EXACT position -- not a global ban on the action. Recorded
        # independent of whether avatar_deltas has an entry for the action
        # yet: an action that never once succeeds never gets a delta
        # recorded, and would otherwise be invisible to block detection
        # (Bug #3).
        self.blocked_edges: Set[Tuple[Tuple[int, int], int]] = set()

    def update_volatility(self, prev: np.ndarray, curr: np.ndarray) -> None:
        diff = prev != curr
        self.change_count += diff.astype(np.int64)
        self.n_transitions += 1
        if self.n_transitions >= MIN_TRANSITIONS_FOR_MASK and self.n_transitions % 5 == 0:
            freq = self.change_count / self.n_transitions
            self.volatile_mask = freq >= VOLATILE_CELL_THRESH

    def masked_hash(self, frame: np.ndarray) -> str:
        masked = np.where(self.volatile_mask, -1, frame)
        return hashlib.md5(masked.tobytes()).hexdigest()

    def blobs_for(self, frame: np.ndarray) -> List[dict]:
        return find_blobs(frame, self.volatile_mask)

    def observe_transition(
        self, prev: np.ndarray, action_id: int, curr: np.ndarray,
        prev_blobs: List[dict], curr_blobs: List[dict],
    ) -> None:
        self.update_volatility(prev, curr)
        for m in match_blob_deltas(prev_blobs, curr_blobs):
            if m["dx"] == 0 and m["dy"] == 0:
                continue
            key = (m["color"], m["size"])
            bucket = self.blob_deltas_by_action.setdefault(action_id, {})
            bucket.setdefault(key, Counter())[(m["dx"], m["dy"])] += 1
        self._maybe_identify_avatar()

    def _maybe_identify_avatar(self) -> None:
        if len(self.blob_deltas_by_action) < 2:
            return
        per_key_action_delta: Dict[Tuple[int, int], Dict[int, Tuple[int, int]]] = {}
        for action_id, key_map in self.blob_deltas_by_action.items():
            for key, counter in key_map.items():
                dom_delta, _ = counter.most_common(1)[0]
                per_key_action_delta.setdefault(key, {})[action_id] = dom_delta
        best_key, best_score = None, 0
        for key, action_delta in per_key_action_delta.items():
            if len(action_delta) < 2 or len(set(action_delta.values())) < 2:
                continue  # same delta regardless of action -> autonomous, not avatar
            score = len(action_delta)
            if score > best_score:
                best_key, best_score = key, score
        if best_key is not None:
            self.avatar_key = best_key
            self.avatar_deltas = dict(per_key_action_delta[best_key])

    def avatar_position(
        self, blobs: List[dict], last_pos: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        if self.avatar_key is None:
            return None
        color, size = self.avatar_key
        candidates = [b for b in blobs if b["color"] == color and b["size"] == size]
        if not candidates:
            return None
        if len(candidates) == 1 or last_pos is None:
            b = candidates[0]
            return (int(round(b["cx"])), int(round(b["cy"])))
        best = min(candidates, key=lambda b: abs(b["cx"] - last_pos[0]) + abs(b["cy"] - last_pos[1]))
        return (int(round(best["cx"])), int(round(best["cy"])))

    def observe_movement_attempt(
        self, frame_before: np.ndarray, action_id: int, pos_before: Tuple[int, int],
        pos_after: Optional[Tuple[int, int]],
    ) -> None:
        if pos_after != pos_before:
            return  # moved -- not blocked, nothing to learn here
        self.blocked_edges.add((pos_before, action_id))
        if action_id not in self.avatar_deltas:
            return  # no known direction -- can't also attribute a wall color
        dx, dy = self.avatar_deltas[action_id]
        step_x = (dx > 0) - (dx < 0)
        step_y = (dy > 0) - (dy < 0)
        tx, ty = pos_before[0] + step_x, pos_before[1] + step_y
        if 0 <= tx < GRID and 0 <= ty < GRID:
            blocking_color = int(frame_before[ty, tx])
            if self.avatar_key is None or blocking_color != self.avatar_key[0]:
                self.wall_colors.add(blocking_color)

    def rare_color_targets(self, frame: np.ndarray, blobs: Optional[List[dict]] = None) -> List[Tuple[int, int]]:
        """One target per connected BLOB (its centroid), not one per cell
        (Bug #4)."""
        total = frame.size
        blobs = blobs if blobs is not None else self.blobs_for(frame)
        excluded = self.wall_colors | ({self.avatar_key[0]} if self.avatar_key is not None else set())
        targets = []
        for b in blobs:
            if b["color"] in excluded:
                continue
            if b["size"] / total <= RARE_COLOR_MAX_FRAC:
                targets.append((int(round(b["cx"])), int(round(b["cy"]))))
        return targets


def bfs_path(
    start: Tuple[int, int], targets: List[Tuple[int, int]],
    deltas: Dict[int, Tuple[int, int]], wall_colors: Set[int], frame: np.ndarray,
    blocked_edges: Optional[Set[Tuple[Tuple[int, int], int]]] = None,
) -> Optional[List[int]]:
    if not targets or not deltas:
        return None
    blocked_edges = blocked_edges or set()
    target_set = set(targets)
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) in target_set and path:
            return path
        for action_id, (dx, dy) in deltas.items():
            if ((x, y), action_id) in blocked_edges:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID and 0 <= ny < GRID) or (nx, ny) in visited:
                continue
            if int(frame[ny, nx]) in wall_colors:
                continue
            visited.add((nx, ny))
            queue.append(((nx, ny), path + [action_id]))
        if len(visited) > 2000:
            break
    return None


class MyAgent(Agent):
    """Stage 5: object-aware navigation on top of Stage 1-4 exploration.

    Each turn:
    1. Update perception (volatility mask, per-blob avatar/wall learning)
       from what our last action actually did.
    2. If the avatar and >=2 of its movement deltas are known, BFS toward
       the nearest not-yet-visited rare-colored blob centroid.
    3. Otherwise fall back to change-rate + state-graph-frontier +
       learned-path-value exploration (masked-hash based) to keep gathering
       data until the avatar can be identified.
    4. For the coordinate action (ACTION6), prefer clicking rare-colored
       cells; track exact (cell, color-shown) no-ops.
    """

    MAX_ACTIONS = 80
    EPSILON = 0.15
    STUCK_EPSILON = 0.5
    NOOP_BAN_THRESHOLD = 4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        self.perception = Perception()

        # Stage 2-4 fallback bookkeeping (masked-hash keyed)
        self.action_stats: Dict[int, List[int]] = {}
        self.graph: Dict[str, Dict[int, str]] = {}
        self.tried_from: Dict[str, Set[int]] = {}
        self.path_value: Dict[Tuple[str, int], float] = {}
        self.episode_path: List[Tuple[str, int]] = []
        self.prev_levels_completed: int = 0

        self.click_noop: Dict[Tuple[int, int, int], int] = {}

        # Every cell the avatar has physically stood on. Fixes a confirmed
        # orbit bug: reaching a target occludes it (the avatar's own sprite
        # covers that cell), so it vanishes from rare_color_targets() the
        # instant we arrive; excluding visited cells from future targets
        # stops the agent snapping right back (Bug #4, same fix family).
        self.visited_positions: Set[Tuple[int, int]] = set()

        self.last_frame: Optional[np.ndarray] = None
        self.last_blobs: Optional[List[dict]] = None
        self.last_hash: Optional[str] = None
        self.last_avatar_pos: Optional[Tuple[int, int]] = None
        self.pending_id: Optional[int] = None
        self.pending_region: Optional[Tuple[int, int]] = None
        self.pending_click: Optional[Tuple[int, int, int]] = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _update_fallback_bookkeeping(self, current_hash: str, levels_completed: int) -> None:
        if self.pending_id is None or self.last_hash is None:
            return
        changed = current_hash != self.last_hash
        stats = self.action_stats.setdefault(self.pending_id, [0, 0])
        stats[0] += 1
        stats[1] += int(changed)

        self.graph.setdefault(self.last_hash, {})[self.pending_id] = current_hash
        self.tried_from.setdefault(self.last_hash, set()).add(self.pending_id)
        self.episode_path.append((self.last_hash, self.pending_id))

        if levels_completed > self.prev_levels_completed:
            for pair in self.episode_path:
                self.path_value[pair] = self.path_value.get(pair, 0.0) + 1.0
            self.episode_path = []
        self.prev_levels_completed = levels_completed

        if self.pending_click is not None:
            key = self.pending_click
            if changed:
                self.click_noop.pop(key, None)
            else:
                self.click_noop[key] = self.click_noop.get(key, 0) + 1

    def _score(self, stats_dict: dict, key: Any) -> float:
        tried, changed = stats_dict.get(key, [0, 0])
        return 1.0 if tried == 0 else changed / tried

    def _pick_by_score(self, stats_dict: dict, keys: list, epsilon: float) -> Any:
        if random.random() < epsilon:
            return random.choice(keys)
        best = max(self._score(stats_dict, k) for k in keys)
        return random.choice([k for k in keys if self._score(stats_dict, k) == best])

    def _fallback_pick_action(self, node_hash: str, legal_ids: list) -> int:
        valued = [(a, self.path_value[(node_hash, a)]) for a in legal_ids
                  if self.path_value.get((node_hash, a), 0.0) > 0]
        if valued:
            best_val = max(v for _, v in valued)
            return random.choice([a for a, v in valued if v == best_val])
        tried_here = self.tried_from.get(node_hash, set())
        untried = [a for a in legal_ids if a not in tried_here]
        if untried:
            return random.choice(untried)
        return self._pick_by_score(self.action_stats, legal_ids, self.STUCK_EPSILON)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.last_frame = None
            self.last_blobs = None
            self.last_hash = None
            self.last_avatar_pos = None
            self.pending_id = None
            self.pending_region = None
            self.pending_click = None
            self.episode_path = []
            self.visited_positions = set()
            return GameAction.RESET

        current_frame = np.array(latest_frame.frame[-1])
        current_blobs = self.perception.blobs_for(current_frame)

        if self.last_frame is not None and self.pending_id is not None and self.last_blobs is not None:
            self.perception.observe_transition(
                self.last_frame, self.pending_id, current_frame, self.last_blobs, current_blobs
            )
            if self.last_avatar_pos is not None:
                pos_after = self.perception.avatar_position(current_blobs, self.last_avatar_pos)
                self.perception.observe_movement_attempt(
                    self.last_frame, self.pending_id, self.last_avatar_pos, pos_after
                )

        current_hash = self.perception.masked_hash(current_frame)
        self._update_fallback_bookkeeping(current_hash, getattr(latest_frame, "levels_completed", 0) or 0)

        legal_ids = [a for a in latest_frame.available_actions if a != GameAction.RESET.value]
        if not legal_ids:
            legal_ids = [a.value for a in GameAction if a is not GameAction.RESET]

        chosen_id: Optional[int] = None
        avatar_pos = self.perception.avatar_position(current_blobs, self.last_avatar_pos)
        if avatar_pos is not None:
            self.visited_positions.add(avatar_pos)

        debug_target = debug_path_len = None
        if avatar_pos is not None and len(self.perception.avatar_deltas) >= 2:
            targets = [t for t in self.perception.rare_color_targets(current_frame, current_blobs)
                       if t not in self.visited_positions]
            path = bfs_path(
                avatar_pos, targets, self.perception.avatar_deltas,
                self.perception.wall_colors, current_frame,
                blocked_edges=self.perception.blocked_edges,
            )
            if path and path[0] in legal_ids:
                chosen_id = path[0]
                debug_path_len = len(path)

        if chosen_id is None:
            chosen_id = self._fallback_pick_action(current_hash, legal_ids)

        if DEBUG:
            print(
                f"[t={self.action_counter}] avatar_key={self.perception.avatar_key} "
                f"avatar_pos={avatar_pos} n_deltas={len(self.perception.avatar_deltas)} "
                f"walls={sorted(self.perception.wall_colors)} chosen={chosen_id} "
                f"path_len={debug_path_len} n_blocked_edges={len(self.perception.blocked_edges)}",
                file=sys.stderr, flush=True,
            )

        action = GameAction.from_id(chosen_id)
        self.pending_id = chosen_id
        self.pending_click = None

        if action.is_simple():
            self.pending_region = None
        elif action.is_complex():
            rare = self.perception.rare_color_targets(current_frame, current_blobs)
            candidates = [
                (x, y) for (x, y) in rare
                if self.click_noop.get((x, y, int(current_frame[y, x])), 0) < self.NOOP_BAN_THRESHOLD
            ]
            if candidates:
                x, y = random.choice(candidates)
            else:
                n_regions = GRID // REGION_SIZE
                region = random.choice([(rx, ry) for rx in range(n_regions) for ry in range(n_regions)])
                x = region[0] * REGION_SIZE + random.randint(0, REGION_SIZE - 1)
                y = region[1] * REGION_SIZE + random.randint(0, REGION_SIZE - 1)
            action.set_data({"x": x, "y": y})
            color_before = int(current_frame[y, x])
            self.pending_click = (x, y, color_before)

        self.last_frame = current_frame
        self.last_blobs = current_blobs
        self.last_hash = current_hash
        self.last_avatar_pos = avatar_pos
        return action
