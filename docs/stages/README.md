# Stage-by-stage build history

Historical snapshots of the agent's code at each stage, kept as real
artifacts rather than prose summaries — each file here is the actual code
that was run at that point, not a reconstruction.

| File | Stage | What it does |
|---|---|---|
| `stage1_available_actions.ipynb` | 1 | Random agent restricted to `available_actions` — the first version that stopped wasting turns on illegal moves. |
| `stage2_change_rate.ipynb` | 2 | Tracks which actions actually change the frame (change-rate per action), biasing choice toward actions known to do something. |
| `stage3_bfs_frontier.ipynb` | 3 | Builds a real state graph from play (`state -> action -> state'`) and prefers actions not yet tried from the current state (BFS frontier). |
| `stage4_learned_path_value.ipynb` | 4 | Monte-Carlo credit assignment: `(state, action)` pairs that preceded `levels_completed` increasing get a learned value, reused across in-episode resets. |
| `stage5a_object_aware_buggy.py` | 5a (superseded) | **The original, buggy** object-aware navigation attempt — avatar position computed as the centroid of every cell matching the avatar's color. This is the version that caused Bug #2 in [`../FINDINGS.md`](../FINDINGS.md) (one color, five unrelated objects, including the floor, averaged into one meaningless centroid). Kept here deliberately as a "what not to do" artifact, not deleted, because it's the direct evidence behind that bug's writeup. |
| `stage5b_object_aware_fixed.py` | 5b (final) | The fixed cumulative core logic: per-blob avatar tracking, position-specific blocked edges, blob-centroid targeting with visited-cell exclusion — i.e. everything in Bugs #2, #3, #4, and #4b's fixes, combined. This is functionally what [`../../agent/my_agent.py`](../../agent/my_agent.py)'s `Perception` class is built from. The three sub-fixes (5b/5c/5d) were applied in place to this one file over the course of debugging rather than kept as separate snapshots — the incremental diff between them is narrated bug-by-bug in FINDINGS.md. |

## Stage-specific test files

The test suite also evolved stage by stage. These are the actual test
files run at each point — not the final consolidated suite
(`tests/test_perception.py`), but the historical ones that drove each fix:

| File | Tests |
|---|---|
| `test_stage4.py` | Stage 4's learned-path-value credit assignment logic. |
| `test_stage5a_buggy.py` | Tests written against the *buggy* `stage5a_object_aware_buggy.py` — passed at the time, which is exactly why Bug #2 wasn't caught by unit tests alone and needed a real game to surface (see "Setup used for all of this" in FINDINGS.md). |
| `test_stage5b_per_blob_fix.py` | Regression tests for Bug #2's fix (per-blob tracking, equal-size matching). |
| `test_stage5c_blocked_edges_fix.py` | Regression tests for Bug #3's fix (position-specific blocked edges, including `test_blocking_is_position_specific_not_global`). |
| `test_stage5d_blob_centroid_fix.py` | Regression tests for Bugs #4 and #4b's fixes (visited-position exclusion, blob-centroid targeting, including `test_visiting_one_blob_leaves_other_blobs_targetable`). |

## Why both the buggy and fixed versions are kept

Deleting `stage5a_object_aware_buggy.py` once it was replaced would have
been the normal move — but the user's instruction here was to document the
real debugging history, and that bug's entire evidence trail (the data in
[`../FINDINGS.md`](../FINDINGS.md), the diagnostic scripts in
[`../../scripts/diagnostics/`](../../scripts/diagnostics/)) only makes sense
next to the actual buggy code it was diagnosing. The final agent never
imports anything from this directory — `agent/my_agent.py` is the only code
path that ships.
