# AGENTS.md

## Project context

This project is an external IsaacLab project for in-hand manipulation research.

Main project path:
- /home/orion/projects/baseline-5-19/in_hand_baseline

IsaacLab path:
- /home/orion/IsaacLab

Conda env:
- env_isaaclab

Current custom environment:
- Isaac-HandOnly-Inspire-Baseline-v0

The current baseline keeps the official G1 + Inspire hand asset but exposes only 24 Inspire hand joint actions. It intentionally avoids Pink IK and the G1 upper-body controller.

## Important rules

- Do not modify files under /home/orion/IsaacLab unless explicitly asked.
- Modify only this external project unless the user approves otherwise.
- Do not run pip install, conda install, apt install, or git checkout without asking first.
- Do not delete large directories or generated assets without asking first.
- In IsaacLab scripts, instantiate AppLauncher before importing isaaclab_tasks, isaaclab.envs, pxr, or omni modules.
- Avoid reintroducing Pink IK into Isaac-HandOnly-Inspire-Baseline-v0.
- Keep the action space for the first baseline at 24 hand joint actions.
- Use short scripts under scripts/ or tools/ rather than editing temporary /tmp scripts when possible.

## Useful commands

Install this external project:
```bash
cd /home/orion/projects/baseline-5-19/in_hand_baseline
python -m pip install -e .
