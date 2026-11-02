# Inspect Robots adapter for Robocurve Arena scenes

This integration uses the existing Arena environment in the same process. It does not install dependencies, modify scene files, touch the teammate's adapter, or create a second SimulationApp.

## Verified compatibility and remaining validation

Local tests use the exact reported server SDK versions, built from Inspect Robots commit `c8a821ba7bd44c042c95ab8f6a83d1e776cfb4a3`:

- `inspect-robots 0.0.1.dev1+gc8a821ba7`
- `inspect-robots-agent 0.29.0`

15 tests pass with fake simulator data, including the real agent with a mocked OpenRouter response and the real Inspect eval/logging loop. No paid calls occur in these tests.

**The Arena bootstrap, live camera rendering, and terminal recorder hook still require a server GPU run.** A broken scene still needs to be fixed in that scene. This integration does not make a successful manipulation policy or an IK planner.

## First run: real camera capture, no model calls

Stop your own existing simulation first. Use the existing Arena interpreter:

```bash
cd /home/ubuntu/workspace/robocurve-scenes
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/run_inspect.py --scene chess-mate --mode capture
```

The default mode enables the three actual sensor cameras, holds the measured robot pose for two steps, saves initial/latest sensor images and state, and exits. No API key is needed.

Outputs have unique directories under `integrations/inspect-arena/outputs/`:

- `run.json`: code revision, dirty file names, versions, scene and mode.
- `initial/` and `latest/`: `overhead.png`, `front.png`, `side.png`, a contact sheet, `cameras.html`, and `capture.json`.

Review these camera images before model control. The viewport streamed to the desktop client is separate from these sensor observations. Image timestamps currently identify the observation's simulation time, not an independently verified camera exposure time.

Select `--scene battery-remote` after its startup fixes work. `--scene atb1-nut` requires that environment to be present on the server and pass startup. Use one scene per fresh process because scene packages share the `my_package` name.

## Stream a held scene

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/run_inspect.py --scene chess-mate \
  --mode hold --steps 1000000 --stream --public-ip 160.211.46.91
```

Client: server `global.prd.ga.run.brev.nvidia.com`, signal `24317`, stream `28337`. A task termination or timeout ends this hold run rather than silently starting another trial. Ctrl+C stops your process. Only one process can use this stream port pair at once.

## Short Argon trial, after camera/state validation

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/run_inspect.py --scene chess-mate \
  --mode agent --steps 50 --max-llm-calls 3
```

This is actual model-driven simulated motion. It uses `google/gemini-4-argon`, explicitly routed to `https://openrouter.ai/api/v1`, the `OPENROUTER_API_KEY` variable, and the Chat Completions wire. If the key is absent it prompts in an interactive terminal with hidden input and does not save it. Do not put the key into command-line arguments or source.

All three RGB images are sent on each policy decision. The agent has a three-call budget by default, at most 2048 output tokens per request, and one HTTP attempt per completion. The limit on environment steps can truncate a motion chunk. The result is a bring-up trial, not the full task benchmark.

Inspect logs, frame sidecars and agent transcripts go into the run's `inspect/` directory. The default upstream policy retains its provider-specific tool metadata between calls.

## Action and observation contract

The outward action has 14 explicitly named **absolute** targets:

```text
left_joint1 ... left_joint6, left_gripper_open,
right_joint1 ... right_joint6, right_gripper_open
```

Joint values are radians in each joint's coordinate frame. Each gripper opening is independent: 0 closed, 1 open. Each controls two physical prismatic finger joints, so there are 16 physical joints.

The adapter inverts the scene-specific mapping rather than passing absolute targets into a home-relative controller:

- Battery/chess arm: `(absolute_target - home) / arm_action_scale`.
- Battery/chess grip: `(desired_opening - home_open_fraction) / gripper_action_scale`.
- Nut arm: `absolute_target - home`.
- Nut grip: `1 - desired_opening` (its native positive command closes).

Joint bounds come from the live articulation's soft limits. Missing bounds or joints fail immediately. Targets must be finite, correctly shaped, in bounds, and change by no more than 0.04 rad per joint / 0.05 opening per environment step. These are command limits, not a collision-free motion guarantee.

Proprioception contains one labeled mixed-unit `joint_command_state` vector matching all 14 actions, plus separate 12-angle `joint_pos`, two-opening `gripper`, and four-position `finger_joint_pos` fields. It is measured absolute state, not `joint_pos_rel` relabeled as absolute. The mixed field's unit is explicitly `rad+normalized`.

Camera labels always map to `overhead`, `front`, `side`. RGB/RGBA input is copied to uint8 RGB without retaining reused simulator buffers. Missing images, unexpected sizes/dtypes, and invalid measured state fail instead of fabricating observations.

## Scoring and reset behavior

The launcher adds an isolated recorder to capture images, state and task reports **before** Arena automatically resets a terminal environment. Terminal results without that capture fail rather than grading the new scene. An explicit reset is required before stepping again after termination.

The task's 0–100 score is preserved as `arena_task_score`. Success requires a success score with no hard failure and a task termination. Failure and timeout remain distinct. A model's `done` claim does not establish success. Policy observations contain only cameras/proprioception; privileged task reports are restricted to scoring/log outputs.

Control rate comes from actual `env.step_dt`. Time budgets currently use **simulation time**; model latency does not advance physics. Agree on wall-clock versus simulation-time benchmark rules before reporting benchmark results.

## Tests

In an environment with the exact Inspect packages and numpy/httpx available:

```bash
python -m unittest discover -s integrations/inspect-arena/tests -v
```

No Isaac imports, GPU, API key, or external model call is needed. Temporary test artifacts are isolated from scene outputs. The tests cover both grip conventions, independent grippers, home-relative inversion, bounds/step rejection, copied/labeled images, terminal capture before auto-reset, partial credit, task failure, ownership cleanup, and the real Inspect/agent interfaces.

## References

- https://github.com/robocurve/inspect-robots
- https://openrouter.ai/google/gemini-4-argon
- https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/
