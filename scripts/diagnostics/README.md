# Diagnostic scripts

These are the actual scripts that produced the evidence quoted in
[`../../docs/FINDINGS.md`](../../docs/FINDINGS.md) — run against real
recorded frames/live sessions from `arcprize.org`, not synthetic data.
None of them are part of the agent; they're kept because "verify with
real evidence before fixing" was the whole workflow, and these are that
evidence, not just the narrative describing it.

| Script | Found |
|---|---|
| `analyze_volatility.py` | Bug #1: diffed 42 real consecutive frames from `ls20`, counted per-cell change frequency, and showed 75 flickering cells (1.8% of the grid) were enough to make the whole-frame hash unique on 41/41 transitions — defeating Stage 3's state-deduplication. |
| `check_blobs.py` | Bug #2's root cause: counted connected components per color per frame and found color 9 was 5 disconnected blobs (sizes 1, 4, 5, 15, 20) on 80/81 real frames, not one object. |
| `check_color4.py` | Bug #2's concrete symptom: proved color 4 — the actual floor, 63% of the grid — was being wrongly learned as a wall, by directly checking what color the centroid-averaging logic was producing. |
| `replay_check.py` | Used to offline-replay real recorded game sessions against the per-blob fix *before* redeploying live, confirming the fix against history rather than only against new live runs. |

Each script expects a recorded/live frame source from a real
ARC-AGI-3-Agents run (see [`../../README.md`](../../README.md) for how to
set that up); they were run ad hoc against `recordings/` output and
aren't wired into the test suite or CI.
