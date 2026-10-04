# Raw debug-run logs

Six raw stderr logs from live runs against the real `ls20` game on
`arcprize.org`, with `ARC3_AGENT_DEBUG=1` set, captured in the order the
fixes in [`../FINDINGS.md`](../FINDINGS.md) were applied. Kept as primary
evidence rather than summarized, since several of the bugs were only
visible by reading the actual per-turn output.

| File | Point in the debugging sequence |
|---|---|
| `run1_ls20_pre_wall_fix.log` | Before Bug #3's fix — shows the 60-action freeze on a single always-blocked action. |
| `run2_ls20_post_wall_fix.log` | After Bug #3's fix (position-specific blocked edges) — freeze resolved. |
| `run3_ls20_per_blob_fix.log` | After Bug #2's fix (per-blob avatar tracking) was deployed live. |
| `run4_ls20_blocked_edges_fix.log` | Follow-up run confirming blocked-edge learning stays scoped per-position. |
| `run5_ls20_visited_positions_fix.log` | After Bug #4's fix — the orbit-loop log referenced directly in FINDINGS.md (`t=32..t=36` sequence). |
| `run6_ls20_blob_centroid_fix.log` | After Bug #4b's fix (one target per blob centroid, not per cell) — most recent run. |
