# LLM-harness build (Duck harness / TAAF)

A second, parallel approach to the classical agent in [`../agent/`](../agent/).

## Why this exists

Research into the current ARC-AGI-3 leaderboard (see [`docs/COMPETITIVE_LANDSCAPE.md`](docs/COMPETITIVE_LANDSCAPE.md))
showed that every top scorer — the Milestone 1 winner and all three Milestone 2
winners — uses the same architecture family: an open-weight LLM wrapped in a
Python-sandbox "agentic coding" harness (Tufa Labs' open-sourced **Duck
harness**, built on their **TAAF** framework), not classical search or
heuristics. Teams differentiate almost entirely on GPU-serving engineering
(quantization, speculative decoding, context management), not game-playing
algorithm design. The classical object-aware BFS agent in `../agent/` is a
different, much weaker category by comparison — it has no mechanism for
general reasoning on novel games the way an LLM does.

Given that, this directory is us building our own competition entry on top
of the Duck harness, following the same pattern every winning team used.

## What's here

```
notebook/
  arc3-duck-harness-smoke-test.ipynb   pulled directly from the Kaggle kernel
                                        (mohamadaiashmar/arc3-seif-attach-test,
                                        version 7) -- the authoritative,
                                        currently-pushed state, not a local
                                        reconstruction
  kernel-metadata.json                 the exact, working push config:
                                        dataset/competition/model sources,
                                        GPU/machine-shape request
docs/
  FINDINGS.md                          the debugging story: every real bug
                                        found getting this notebook to
                                        actually mount data and run, in order,
                                        with evidence -- same format as
                                        ../../docs/FINDINGS.md
  PUBLIC_RESOURCES.md                  the public Kaggle datasets/models this
                                        build depends on, with verified sizes,
                                        licenses, and what each one actually
                                        contains
```

## Status

Confirmed working end-to-end, on CPU: competition data mounts, both datasets
mount (Pennyroyal SGLang wheelhouse, TAAF source bundle), both models mount,
and the full offline Pennyroyal/SGLang install completes successfully against
a Python-3.12 venv (the account's Kaggle image runs 3.13 by default, which is
ABI-incompatible with the wheelhouse's pinned deps -- see FINDINGS.md Bug #4).

**Not yet confirmed**: actually serving the model and playing a game on the
intended `NvidiaRtxPro6000` accelerator. Three separate isolation tests (see
FINDINGS.md's final section) ruled out account eligibility (real GPU quota
confirmed via API, 30h/week, barely used), the push mechanism (a plain
`NvidiaTeslaT4` request on the same path provisions a real GPU immediately),
the accelerator string (byte-identical to a real working competitor
submission's metadata), and stale kernel state (fails identically on a
brand-new kernel). What's left is accelerator-pool capacity: the RTX Pro
6000 request appears to silently fall back to CPU when that specific pool
has no capacity, rather than erroring or queuing -- consistent with other
competitors independently reporting multi-hour queues for this exact
accelerator on the competition forum. Next: retry periodically, and
separately evaluate whether a T4 fallback is viable (its 15 GB VRAM can't
fit the current 181 GB quantized model as-is, so this would need a much
smaller model, not a drop-in swap).

## Reproducing / continuing this work

```bash
# Push the exact working config back to Kaggle (overwrites whatever's live
# at that kernel slug with this repo's checked-in version):
kaggle kernels push -p llm-harness/notebook
```

This assumes the real `kaggle` CLI with valid credentials
(`~/.kaggle/kaggle.json`) -- the MCP tool wrappers used to build this
(`seif-kaggle`'s `kernel_push`, which reads the same `kernel-metadata.json`
format) aren't guaranteed to be available outside this session.
