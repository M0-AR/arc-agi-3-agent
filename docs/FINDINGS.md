# Findings: four real bugs, found with evidence, fixed one at a time

This document is the actual debugging story behind this agent. Every bug
below was found by looking at real data (live API responses, recorded
frames, or direct replay of past games) — none were guessed, and none were
"fixed" without a regression test proving the fix and the earlier behavior
it replaces.

The agent was built in five incremental stages. Stages 1–4 are blind
exploration strategies (respect legal actions, track which actions change
anything, build a state graph and prefer unexplored edges, learn which
state→action pairs advanced a level). Stage 5 adds object-aware
navigation — and that's where all four bugs below were found, each one by
running the agent against the real ARC-AGI-3 service and reading the
evidence rather than assuming the first plausible-looking implementation
was correct.

## Setup used for all of this

A local checkout of [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents),
with this repo's `agent/my_agent.py` dropped into its `agents/templates/`
folder, pointed at the real `arcprize.org` API with a free developer key
(see `../README.md` for exact steps). This matters: none of these bugs were
visible from unit tests alone. Every one needed a real game.

The actual code at each build stage is preserved in [`stages/`](stages/)
(not just described here), the scripts that produced the evidence below
are in [`../scripts/diagnostics/`](../scripts/diagnostics/), and the raw
live-run logs are in [`logs/`](logs/).

---

## Bug #1 — background animation poisoned the state-deduplication hash

**Symptom:** Stage 3's "prefer an action not yet tried from this state"
logic never once recognized a repeated state. Every single frame hashed as
brand new.

**Evidence:** Recorded 42 real frames from `ls20`, diffed every consecutive
pair, and counted how often each cell changed:

```
whole-frame hash changed on 41/41 transitions (100%)
cells changing in >=20% of transitions: 75/4096 (1.8%)
distinct whole-frame states: 42/42
```

75 cells, clustered at one spot on the screen, were flickering between two
colors on almost every frame regardless of what action was taken — a
decorative animation or counter. Because the hash covered the *whole*
frame, that one small flickering patch alone was enough to make every
frame's hash unique, even when the player hadn't moved at all.

**Fix:** track each cell's change-frequency across the episode; once a cell
changes in ≥20% of observed transitions, exclude it from the hash. The
first 10 transitions run unmasked (not enough data yet), then the mask
recomputes every 5 transitions.

## Bug #2 — one color, five unrelated objects

**Symptom:** after fixing Bug #1 and adding BFS navigation, the agent
"learned" walls that were provably wrong — including the actual floor
color, which covered 63% of the grid.

**Evidence:** walked back from the wrong wall to its cause. Avatar position
was being computed as the centroid of *every* cell matching the avatar's
color. Checked directly:

```python
# color 9 in ls20, counted per frame:
blob-count distribution=[0, 5], frames with >1 blob: 80/81
example blob sizes: [5, 1, 15, 20, 4]
```

Color 9 was never one object — it was five disconnected shapes (sizes 1,
4, 5, 15, 20 cells) on almost every frame. Replaying real recordings
confirmed only the size-15 blob ever actually moved; the rest were static
HUD/decoration. Averaging all five into one centroid produced a position
that didn't correspond to anything real, and "it didn't move" readings
were really just the big static blobs outweighing the one that did.

**Fix:** track movement **per connected blob** (color + size, matched
frame-to-frame by nearest centroid), not per color. Requiring equal size
when matching is what prevents a moving object from ever being confused
with a differently-sized static blob of the same color.

## Bug #3 — an action that never succeeds is invisible to its own diagnosis

**Symptom:** even after Bug #2's fix, the avatar still froze completely for
60 straight actions, repeatedly choosing the same action with no effect,
and no wall was ever learned to explain why.

**Evidence:** traced the logic directly. Wall-detection was gated on
`action_id in self.avatar_deltas` — but deltas are only ever recorded for
transitions with *actual movement* (`dx != 0 or dy != 0`). An action that
is **always** blocked therefore never produces a delta, never enters
`avatar_deltas`, and the wall-detection check returns immediately without
learning anything. The one failure mode most worth diagnosing was the one
the diagnosis code couldn't see.

Confirmed this matches standard practice before fixing it — online
grid-pathfinding literature uses exactly this approach: "remove the
observed blocked edge" from the search graph, as a property of the
specific *(position, action)* pair, not a global property of the action.

**Fix:** record `(position, action_id)` as a blocked edge the moment
movement fails, independent of whether a delta is known for that action.
BFS excludes known-blocked edges from its search. Verified this is
precisely scoped — blocking `(10,10)→action 3` does not block
`action 3` from any other position (see
`test_blocking_is_position_specific_not_global`).

## Bug #4 — reaching a target makes it vanish, so the agent orbits it forever

**Symptom:** after Bug #3's fix, the avatar moved freely across large
distances — but still completed zero levels. Debug logging of the actual
BFS target each turn showed the same target being reached, then
immediately re-targeted, over and over:

```
t=32 avatar_pos=(21,38) target=(21,33) path_len=1
t=33 avatar_pos=(21,33) target=None      <- arrived
t=34 avatar_pos=(16,33) target=None      <- wandered off (fallback, no target visible)
t=35 avatar_pos=(21,33) target=None      <- back again
t=36 avatar_pos=(26,33) target=None      <- wandered off again
```

**Cause:** when the avatar physically stands on a target cell, that cell's
displayed color becomes the avatar's own color — it's visually occluded.
`rare_color_targets()` reads live frame colors with no memory, so the
instant the agent arrives, the target disappears from the target list.
With nothing to navigate toward, the exploration fallback takes over and
wanders off; one step later the target reappears (no longer occluded), and
BFS snaps right back. This repeated for the rest of the action budget.

**Fix:** track every cell the avatar has physically visited, and exclude
visited cells from future target lists — regardless of what color they
currently show.

## Bug #4b — a found-while-fixing-#4 bug: per-cell targets let one object get "exhausted" by accident

Fixing Bug #4 surfaced a second problem. With visited-cell exclusion in
place, the target list went completely empty by turn 31 — every rare color
had been marked "visited" despite the agent barely having explored.

**Cause:** `rare_color_targets()` returned one target *per matching cell*,
not per object. A single object made of many cells (one color measured at
up to 84 cells, just under the 2%-of-grid rarity cutoff) was being treated
as 84 separate targets. Incidental wandering over a handful of its cells —
without ever reaching anything resembling its center — was enough to mark
the *entire* object exhausted.

Checked this against two independent, unrelated real agent implementations
targeting this same benchmark before fixing it:
[BDR-Pro/arc-prize-2026-arc-agi-3](https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3)
and
[Crispsec/arc-agi-3-object-centric-agent](https://github.com/Crispsec/arc-agi-3-object-centric-agent)
— both use connected-component analysis and target **object centroids**,
not raw cells. Confirms this is the standard approach, not a one-off
guess.

**Fix:** `rare_color_targets()` now returns one target per connected blob
— its centroid — reusing the same blob-finding logic Bug #2's fix already
built. Visiting one object's centroid no longer has any effect on a
separate object elsewhere on the board (`test_visiting_one_blob_leaves_other_blobs_targetable`).

---

## Where this leaves the agent

After all four fixes: the agent reliably identifies the avatar, learns
real walls (not the floor), never gets permanently stuck on a single
action, and explores multiple distinct objects instead of orbiting the
first one it finds. It has not yet completed a full level in testing — the
remaining gap looks like a matter of search efficiency and action budget
(80 actions, much of it spent on necessary wall-discovery detours) rather
than a logic bug, but this has **not** been separately confirmed the way
the four bugs above were, and should be treated as an open question, not a
conclusion.
