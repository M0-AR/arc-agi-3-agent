"""Step 5b: per-blob avatar tracking, replacing the broken per-color-centroid
approach. Root cause that forced this rewrite: colors on real ARC-AGI-3
frames can be shared by multiple disconnected objects (we measured ls20's
color 9 as 5 separate blobs on 80/81 real frames), so averaging all cells of
a color into one centroid mixes a moving object with static decorations of
the same color."""
import hashlib
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

GRID = 64
MIN_TRANSITIONS_FOR_MASK = 10
VOLATILE_CELL_THRESH = 0.20
RARE_COLOR_MAX_FRAC = 0.02


def find_blobs(frame: np.ndarray, ignore_mask: np.ndarray) -> List[dict]:
    """4-connected flood fill. Returns one dict per blob:
    {color, size, cx, cy}. Ignored (volatile) cells are skipped entirely."""
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
            blobs.append({
                "color": color, "size": len(xs),
                "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
            })
    return blobs


def match_blob_deltas(prev_blobs: List[dict], curr_blobs: List[dict]) -> List[dict]:
    """Match same-color, same-size blobs between two frames by nearest
    centroid (handles multiple same-colored blobs: matching requires equal
    size, so distinctly-sized same-colored blobs never get confused with
    each other). Returns per-matched-blob {color, size, dx, dy}."""
    prev_by_color: Dict[int, List[dict]] = defaultdict(list)
    curr_by_color: Dict[int, List[dict]] = defaultdict(list)
    for b in prev_blobs:
        prev_by_color[b["color"]].append(b)
    for b in curr_blobs:
        curr_by_color[b["color"]].append(b)

    results = []
    for color, p_list in prev_by_color.items():
        c_list = curr_by_color.get(color, [])
        used = set()
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
                results.append({
                    "color": color, "size": p["size"],
                    "dx": round(c["cx"] - p["cx"]), "dy": round(c["cy"] - p["cy"]),
                })
    return results


class Perception:
    def __init__(self) -> None:
        self.change_count = np.zeros((GRID, GRID), dtype=np.int64)
        self.n_transitions = 0
        self.volatile_mask = np.zeros((GRID, GRID), dtype=bool)

        # (color, size) -> action_id -> Counter of observed (dx, dy)
        self.blob_deltas_by_action: Dict[Tuple[int, int], Dict[int, Counter]] = {}
        self.avatar_key: Optional[Tuple[int, int]] = None  # (color, size)
        self.avatar_deltas: Dict[int, Tuple[int, int]] = {}
        self.wall_colors: Set[int] = set()
        # (position, action_id) pairs observed to produce zero movement from
        # that EXACT position. Position-specific, not a global ban on the
        # action. Recorded independent of whether avatar_deltas has an entry
        # for the action yet, since an action that NEVER succeeds never gets
        # a delta recorded (zero-movement transitions are skipped) and would
        # otherwise be invisible to block detection forever.
        self.blocked_edges: Set[Tuple[Tuple[int, int], int]] = set()

        self._blob_cache: Dict[int, List[dict]] = {}  # id(frame) -> blobs, tiny LRU of 1

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
        prev_blobs: Optional[List[dict]] = None, curr_blobs: Optional[List[dict]] = None,
    ) -> None:
        self.update_volatility(prev, curr)
        prev_blobs = prev_blobs if prev_blobs is not None else self.blobs_for(prev)
        curr_blobs = curr_blobs if curr_blobs is not None else self.blobs_for(curr)
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
            if len(action_delta) < 2:
                continue
            if len(set(action_delta.values())) < 2:
                continue  # same delta regardless of action -> autonomous
            score = len(action_delta)
            if score > best_score:
                best_key, best_score = key, score
        if best_key is not None:
            self.avatar_key = best_key
            self.avatar_deltas = dict(per_key_action_delta[best_key])

    def avatar_position(
        self, frame: np.ndarray, blobs: Optional[List[dict]] = None,
        last_pos: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        if self.avatar_key is None:
            return None
        color, size = self.avatar_key
        blobs = blobs if blobs is not None else self.blobs_for(frame)
        candidates = [b for b in blobs if b["color"] == color and b["size"] == size]
        if not candidates:
            return None
        if len(candidates) == 1 or last_pos is None:
            b = candidates[0]
            return (int(round(b["cx"])), int(round(b["cy"])))
        # ambiguous: multiple blobs with the avatar's exact (color,size) --
        # pick whichever is nearest to where we last saw it
        best = min(candidates, key=lambda b: abs(b["cx"] - last_pos[0]) + abs(b["cy"] - last_pos[1]))
        return (int(round(best["cx"])), int(round(best["cy"])))

    def observe_movement_attempt(
        self, frame_before: np.ndarray, action_id: int, pos_before: Tuple[int, int],
        pos_after: Optional[Tuple[int, int]],
    ) -> None:
        if pos_after != pos_before:
            return  # moved -- not blocked, nothing to learn here
        # Record the specific (position, action) edge as blocked regardless
        # of whether we have a global delta for this action -- this is the
        # piece that was missing: an action that never once succeeds would
        # otherwise never accumulate a delta and so could never be diagnosed.
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
        """One target per connected BLOB (its centroid), not one per cell.
        Matches how multiple independent 2026 ARC-AGI-3 agents handle this
        (e.g. Crispsec/arc-agi-3-object-centric-agent: "object centroids...
        using connected components"). Fixes a confirmed bug: a single rare-
        colored blob spanning many cells was previously treated as that many
        separate targets, so incidental wandering over a few of its cells
        could exhaust the entire object as "visited" without ever reaching
        its actual center -- and conversely, a large blob dominated the
        target list disproportionately to how many distinct objects exist."""
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
