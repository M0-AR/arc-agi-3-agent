"""Step 5 core logic, extracted standalone so it can be unit-tested with
synthetic frames before ever touching the live environment or Kaggle.
Mirrors exactly what goes into my_agent.py's %%writefile cell.
"""
import hashlib
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

GRID = 64
MIN_TRANSITIONS_FOR_MASK = 10
VOLATILE_CELL_THRESH = 0.20
RARE_COLOR_MAX_FRAC = 0.02  # a color covering <2% of non-background cells is "rare"


class Perception:
    """Everything about understanding the frame: volatility masking, avatar
    detection via diff-based color tracking, wall/pushable classification."""

    def __init__(self) -> None:
        self.change_count = np.zeros((GRID, GRID), dtype=np.int64)
        self.n_transitions = 0
        self.volatile_mask = np.zeros((GRID, GRID), dtype=bool)  # recomputed periodically

        # action_id -> color -> Counter of observed (dx, dy) deltas
        self.color_deltas_by_action: Dict[int, Dict[int, Counter]] = {}
        self.avatar_color: Optional[int] = None
        self.avatar_deltas: Dict[int, Tuple[int, int]] = {}  # action_id -> (dx, dy)
        self.wall_colors: Set[int] = set()
        self.pushable_colors: Set[int] = set()

    # ---------- volatility masking ----------
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

    # ---------- avatar / mover detection ----------
    def _color_centroids(self, frame: np.ndarray, ignore_mask: np.ndarray) -> Dict[int, Tuple[float, float, int]]:
        """color -> (centroid_x, centroid_y, cell_count), over non-ignored cells."""
        out = {}
        for color in np.unique(frame):
            color = int(color)
            cell_mask = (frame == color) & (~ignore_mask)
            ys, xs = np.where(cell_mask)
            if len(xs) == 0:
                continue
            out[color] = (float(xs.mean()), float(ys.mean()), int(len(xs)))
        return out

    def observe_transition(self, prev: np.ndarray, action_id: int, curr: np.ndarray) -> None:
        self.update_volatility(prev, curr)
        ignore = self.volatile_mask
        prev_c = self._color_centroids(prev, ignore)
        curr_c = self._color_centroids(curr, ignore)

        for color, (px, py, pcount) in prev_c.items():
            if color not in curr_c:
                continue
            cx, cy, ccount = curr_c[color]
            if pcount != ccount:
                continue  # shape/size changed -- not a simple mover
            dx, dy = round(cx - px), round(cy - py)
            if dx == 0 and dy == 0:
                continue
            bucket = self.color_deltas_by_action.setdefault(action_id, {})
            bucket.setdefault(color, Counter())[(dx, dy)] += 1

        self._maybe_identify_avatar()

    def _maybe_identify_avatar(self) -> None:
        if self.avatar_color is not None:
            # still refine known deltas
            pass
        # need data from >=2 distinct actions to say anything
        if len(self.color_deltas_by_action) < 2:
            return

        # dominant delta per (action, color)
        per_color_action_delta: Dict[int, Dict[int, Tuple[int, int]]] = {}
        for action_id, color_map in self.color_deltas_by_action.items():
            for color, counter in color_map.items():
                dom_delta, _ = counter.most_common(1)[0]
                per_color_action_delta.setdefault(color, {})[action_id] = dom_delta

        best_color, best_score = None, 0
        for color, action_delta in per_color_action_delta.items():
            if len(action_delta) < 2:
                continue
            distinct_deltas = set(action_delta.values())
            if len(distinct_deltas) < 2:
                continue  # same delta regardless of action -> autonomous, not avatar
            score = len(action_delta)  # more actions explained -> stronger candidate
            if score > best_score:
                best_color, best_score = color, score

        if best_color is not None:
            self.avatar_color = best_color
            self.avatar_deltas = dict(per_color_action_delta[best_color])

    def avatar_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        if self.avatar_color is None:
            return None
        ys, xs = np.where((frame == self.avatar_color) & (~self.volatile_mask))
        if len(xs) == 0:
            return None
        return (int(round(xs.mean())), int(round(ys.mean())))

    # ---------- wall / pushable learning ----------
    def observe_movement_attempt(
        self, frame_before: np.ndarray, action_id: int, pos_before: Tuple[int, int],
        pos_after: Optional[Tuple[int, int]],
    ) -> None:
        """Called when we attempted a *known* avatar-moving action. Only
        treat this as a wall when the avatar did not move AT ALL -- not
        merely "moved somewhere other than my current delta guess", since
        that guess can still be drifting (e.g. 1 vs 2 cells) and a mismatch
        there is not evidence of a wall, just an imprecise estimate."""
        if action_id not in self.avatar_deltas:
            return
        if pos_after != pos_before:
            return  # it moved somewhere -- not blocked, nothing learned here
        dx, dy = self.avatar_deltas[action_id]
        # Use a unit step in the delta's direction, not the (possibly still
        # drifting) full magnitude, to find the immediately-adjacent blocker.
        step_x = (dx > 0) - (dx < 0)
        step_y = (dy > 0) - (dy < 0)
        tx, ty = pos_before[0] + step_x, pos_before[1] + step_y
        if 0 <= tx < GRID and 0 <= ty < GRID:
            blocking_color = int(frame_before[ty, tx])
            if blocking_color != self.avatar_color:
                self.wall_colors.add(blocking_color)

    # ---------- rare-color target detection ----------
    def rare_color_targets(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Cells whose color is uncommon and not background/wall/avatar --
        likely interactive/goal objects, per the objectness prior."""
        total = frame.size
        colors, counts = np.unique(frame, return_counts=True)
        excluded = self.wall_colors | ({self.avatar_color} if self.avatar_color is not None else set())
        rare_colors = [
            int(c) for c, cnt in zip(colors, counts)
            if int(c) not in excluded and (cnt / total) <= RARE_COLOR_MAX_FRAC
        ]
        targets = []
        for c in rare_colors:
            ys, xs = np.where(frame == c)
            targets.extend(zip(xs.tolist(), ys.tolist()))
        return targets


def bfs_path(
    start: Tuple[int, int],
    targets: List[Tuple[int, int]],
    deltas: Dict[int, Tuple[int, int]],
    wall_colors: Set[int],
    frame: np.ndarray,
) -> Optional[List[int]]:
    """BFS over the grid using only the learned avatar movement deltas as
    edges. Returns the list of action_ids to reach the nearest target, or
    None if unreachable."""
    if not targets or not deltas:
        return None
    target_set = set(targets)
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) in target_set and path:
            return path
        for action_id, (dx, dy) in deltas.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                continue
            if (nx, ny) in visited:
                continue
            if int(frame[ny, nx]) in wall_colors:
                continue
            visited.add((nx, ny))
            queue.append(((nx, ny), path + [action_id]))
        if len(visited) > 2000:  # safety cap
            break
    return None
