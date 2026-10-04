"""Builds notebook/arc3-agent-submission.ipynb by embedding agent/my_agent.py
verbatim into a %%writefile cell, inside the same wrapper structure the
official Kaggle sample submission uses (install wheels -> write agent ->
rerun-mode harness glue -> dummy-submission fallback for interactive runs).
Keeping the agent as a single source file edited in one place (agent/) and
generating the notebook from it avoids the two ever drifting apart.
"""
import nbformat as nbf
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_SRC = os.path.join(HERE, "..", "agent", "my_agent.py")

nb = nbf.v4.new_notebook()

with open(AGENT_SRC) as f:
    agent_code = f.read()

md1 = nbf.v4.new_markdown_cell(
    "# ARC-AGI-3 Submission\n\n"
    "Agent source lives in `agent/my_agent.py` in this repo and is embedded "
    "verbatim below. See `docs/FINDINGS.md` for the design rationale and "
    "the bugs found/fixed to get here, and `README.md` for how to test "
    "this agent live against the real ARC-AGI-3 service before submitting."
)

cell_install = (
    "!pip install --no-index --find-links \\\n"
    "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
    "    arc-agi python-dotenv\n"
)

cell_agent = "%%writefile /kaggle/working/my_agent.py\n" + agent_code

cell_rerun = r'''import os

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \
          --retry-max-time 600 http://gateway:8001/api/games

    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
           /kaggle/working/ARC-AGI-3-Agents

    !cp /kaggle/working/my_agent.py \
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write("""from typing import Type, cast
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
    "myagent": MyAgent,
}
""")

    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write("""SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/kaggle/working/server_recording
""")

    !cd /kaggle/working/ARC-AGI-3-Agents && \
        MPLBACKEND=agg \
        python main.py --agent myagent
'''

cell_submission = '''import os
import pandas as pd

if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 1]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
'''

nb['cells'] = [
    md1,
    nbf.v4.new_code_cell(cell_install),
    nbf.v4.new_code_cell(cell_agent),
    nbf.v4.new_code_cell(cell_rerun),
    nbf.v4.new_code_cell(cell_submission),
]

nb['metadata'] = {
    "kaggle": {
        "accelerator": "none",
        "dataSources": [{"sourceType": "competition", "sourceId": "", "isSourceIdPinned": False}],
        "isInternetEnabled": False,
        "language": "python",
        "sourceType": "notebook",
        "isGpuEnabled": False,
    },
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out_path = os.path.join(HERE, "arc3-agent-submission.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print("wrote", out_path)
