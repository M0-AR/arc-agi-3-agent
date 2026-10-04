# ARC-AGI-3 Agent

An object-aware agent for [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3),
built incrementally and verified against the real ARC-AGI-3 service at
every stage — no simulated rollouts, no code borrowed wholesale from other
submissions. Four real bugs were found and fixed along the way, each with
direct evidence and a regression test; the full story is in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## What ARC-AGI-3 actually is

An interactive benchmark where the agent is dropped into a 64×64 grid-based
game with **zero instructions** — no stated goal, no tutorial, 16 possible
colors per cell, 8 possible actions (movement-style buttons plus one
"click a coordinate" action). It has to infer the rules and the goal purely
by acting and observing. Humans solve 100% of these environments; frontier
AI systems scored under 1% as of early 2026. Scoring compares the number of
actions an agent needs against a human baseline, so efficiency — not just
eventual success — is what's measured.

## Architecture

Built in five stages, each one a genuine increment over the last:

1. **Respect `available_actions`** instead of trying illegal moves blindly.
2. **Change-rate tracking** — learn which actions actually affect the frame.
3. **State-graph BFS frontier** — build a real graph from play, prefer
   actions not yet tried from the current state.
4. **Learned path value** — Monte-Carlo credit assignment on
   `(state, action)` pairs that advanced a level, reused across in-episode
   resets.
5. **Object-aware navigation** — once Stage 1–4 bootstraps enough data:
   - **Volatility-masked state hashing**, so background animation/counters
     can't make every frame look "new."
   - **Per-blob avatar detection**: diff consecutive frames, find which
     connected-component (not raw color — color alone was proven to
     sometimes cover multiple unrelated objects) moves consistently per
     action while excluding colors that move the same way regardless of
     input (gravity/conveyors, not the player).
   - **Position-specific blocked-edge learning**: when movement fails,
     record exactly `(position, action)` as blocked, not the action
     globally — matches standard practice in online grid-pathfinding
     literature.
   - **Blob-centroid BFS targeting**: navigate toward the nearest
     not-yet-visited object, treating each connected blob as one target,
     not one target per cell.

Stage 5's components only engage once the avatar is identified; before
that (and whenever no valid target/path exists), the agent falls back to
Stage 2–4's exploration.

## Repo layout

```
agent/my_agent.py          the agent — single file, embedded into the
                            submission notebook unmodified
tests/test_perception.py   11 unit tests, one per confirmed bug + the
                            underlying logic it fixes
notebook/
  build_notebook.py         regenerates the submission notebook from
                             agent/my_agent.py (keeps them from drifting)
  arc3-agent-submission.ipynb   the actual Kaggle submission artifact
docs/FINDINGS.md            the debugging story: what broke, the evidence,
                             the fix, the test that proves it
```

## Running the tests

```bash
pip install numpy pytest
python -m pytest tests/test_perception.py -v
```

These test the perception/navigation logic in isolation against synthetic
frames (no live API, no Kaggle). `agent/my_agent.py` itself (the `MyAgent`
class) depends on the real ARC-AGI-3-Agents runtime and isn't imported
directly by the test suite — see below for testing the full agent.

## Testing the full agent against real games (recommended before submitting)

The agent needs a real ARC-AGI-3 game server to run against. Kaggle's
sandbox only provides one *during an actual scored submission* — so to
iterate without burning submission attempts, run locally against the real
public practice games:

1. Get a free API key: [arcprize.org/platform](https://arcprize.org/platform) →
   log in → profile → API Keys → create one.
2. Clone the agent runtime:
   ```bash
   git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
   cd ARC-AGI-3-Agents
   python -m venv venv && ./venv/bin/pip install arc-agi python-dotenv numpy requests pydantic
   ```
3. Drop this repo's agent in:
   ```bash
   cp /path/to/this/repo/agent/my_agent.py agents/templates/my_agent.py
   ```
4. Write a minimal `agents/__init__.py` (avoids pulling in the heavy
   LLM-agent dependencies this repo's agent doesn't need):
   ```python
   from typing import Type
   from dotenv import load_dotenv
   from .agent import Agent, Playback
   from .swarm import Swarm
   from .templates.random_agent import Random
   from .templates.my_agent import MyAgent
   load_dotenv()
   AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}
   ```
5. `.env`:
   ```
   SCHEME=https
   HOST=arcprize.org
   PORT=443
   ARC_API_KEY=your_key_here
   ARC_BASE_URL=https://arcprize.org
   OPERATION_MODE=online
   RECORDINGS_DIR=recordings
   ```
6. Run it: `./venv/bin/python main.py --agent myagent --game <game_id>`
   (list available games: `curl -H "X-API-Key: $ARC_API_KEY" https://arcprize.org/api/games`)

Set `ARC3_AGENT_DEBUG=1` in the environment to get per-turn perception
state (avatar position, known walls, current BFS target) printed to
stderr.

## Submitting to Kaggle

`notebook/arc3-agent-submission.ipynb` is ready to push as-is — it embeds
`agent/my_agent.py` via a `%%writefile` cell inside the same
install/rerun-mode/dummy-submission wrapper structure the official sample
submission uses. If you change the agent, regenerate the notebook rather
than editing it directly:

```bash
python notebook/build_notebook.py
```

## Status

Reliably identifies the avatar, learns real walls, never permanently
freezes on a blocked action, and explores multiple distinct objects
instead of orbiting the first one found. Has not yet completed a full
level in testing — the current working hypothesis is that this is a
search-efficiency/action-budget issue rather than a remaining logic bug,
but that hypothesis is **not yet separately confirmed** the way the four
fixed bugs were. See the "Where this leaves the agent" section at the end
of `docs/FINDINGS.md`.

## License

MIT — see [LICENSE](LICENSE).
