# Competitive landscape research (what's actually winning)

Research done directly against the live competition (Kaggle leaderboard,
forum threads, and public notebooks) before committing to this architecture.
Everything below is sourced from real, live data pulled via the Kaggle API,
not assumption.

## Competition facts (as of research date, Oct 2026)

- Deadline 2026-11-02, new-entrant cutoff 2026-10-26 (already entered).
- 1 submission/day, code-competition only (submit via Kaggle Notebook).
- Scoring: 100 hidden games, 55 drive the public leaderboard, 55 held back
  for final ranking. LB numbers are literal percentages (confirmed by a
  competition host directly on the forum: "It is 0.33%" not "33%").
- Current public LB leader at research time: Tufa Labs, **55.89%**.

## The architecture that's actually winning

Every top scorer — Milestone 1 winner (Tufa Labs) and all three Milestone 2
winners (dfranzen, lordhansolo, sirikilohit) — runs the **same core
approach**: an open-weight LLM (Qwen3-family) wrapped in a Python-sandbox
"agentic coding" harness, **Tufa Labs' open-sourced "Duck harness"**
(`github.com/Tufalabs/duck-harness`, MIT-licensed), running on their own
**TAAF** (Tufa ARC-AGI Framework) execution layer.

The duck is given the game state as Python variables -- a frame, a
segmentation graph of objects (connected components with shape/rotation-
canonical hashes), diff history -- and it writes Python code each turn to
inspect the board and decide what to do. There is no MCTS, no RL, no
hand-written search anywhere in the winning code.

**What differentiates individual entries is almost entirely GPU-serving
engineering**, not game logic: quantization scheme (AutoRound W4A16,
NVFP4+FP8), speculative decoding, context-trim thresholds, and per-game
token-budget scheduling. Scores climbed from ~1.2% (Milestone 1, June) to
~22-28% (Milestone 2, Sept) to 55.89% (Oct) purely by tuning that serving
stack around the same harness.

| Team | Milestone | Score | Approach |
|---|---|---|---|
| Tufa Labs | 1 (winner) | 1.21% self-reported (0.77-1.6% range) | Original Duck harness, Qwen3.6-27B-FP8 |
| dfranzen | 2 (winner) | 27.89% | Duck harness fork, SGLang Pennyroyal, Qwen3.8-Flash-Next AutoRound W4A16 |
| lordhansolo | 2 (winner) | 23.84% | Duck harness fork, vLLM, NVFP4+FP8 mixed quant |
| sirikilohit | 2 (winner) | 22.53% | Duck harness fork, SGLang Pennyroyal, UCB token scheduler |
| Tufa Labs | (current LB leader) | 55.89% | Further-tuned Duck harness lineage |

## A tempting but misleading comparison point

A public, 153-upvote notebook (`[0.46] ARC-AGI-3: Persistent Memory BFS ARC
Solver`) superficially resembles our own classical object-aware BFS agent
(`../../agent/my_agent.py`) and claims a 0.46 score. It is **not a fair
comparison**: it loads each game's actual Python source file off disk
(`find_game_source` / `importlib`) and runs BFS/A*/MCTS against a free,
in-process deepcopy'd simulator -- white-box access to the game's full
source that is not available against the real, hidden competition games over
the live API. Its score is not evidence that a legitimate live-API-only
classical agent can reach that level.

## Why we chose to build on Duck harness rather than keep extending the
classical agent

Our classical agent (`../../agent/`) is real, validated, original work -- but
it is a fundamentally different and much weaker category of approach. It has
no mechanism for general reasoning on a novel game the way an LLM does, so it
cannot close the gap to the current leaderboard no matter how much it's
tuned. Given the user's explicit goal of the best possible result, and given
every serious competitor has converged on the same architecture, building on
the proven, open-sourced Duck harness -- the same thing every Milestone 2
winner did -- is the direct, evidence-backed path to a competitive score,
rather than re-deriving GPU-serving infrastructure that the community has
already spent months tuning.

## Public resources that make this reproducible without Tufa-private access

See [`PUBLIC_RESOURCES.md`](PUBLIC_RESOURCES.md) for the exact datasets and
models (wheelhouse, source bundle, quantized model, draft model) that let us
reproduce dfranzen's Milestone 2 setup end-to-end using only public Kaggle
resources -- no access to Tufa Labs' or dfranzen's private infrastructure
required.
