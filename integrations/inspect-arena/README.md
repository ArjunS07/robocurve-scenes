# Inspect Robots adapter for Robocurve Arena scenes

This integration uses the existing Arena environment in the same process. It does not install dependencies, modify scene files, touch the teammate's adapter, or create a second SimulationApp.

## Verified compatibility and remaining validation

Local tests use the exact reported server SDK versions, built from Inspect Robots commit `c8a821ba7bd44c042c95ab8f6a83d1e776cfb4a3`:

- `inspect-robots 0.0.1.dev1+gc8a821ba7`
- `inspect-robots-agent 0.29.0`

45 tests pass with fake simulator data, including the real agent with a mocked OpenRouter response and the real Inspect eval/logging loop. No paid calls occur in these tests.

The server has verified the scene/camera bootstrap, Argon communication, raw-joint motion, and MP4 export. The server reach check passed all nine small-motion phases with final position errors below 0.1 mm, and consecutive agent moves now execute. Full piece grasping and reliable visual board interpretation are still unproven. A broken scene still needs to be fixed in that scene. Passing the reach check establishes local tracking, not a successful piece grasp or chess benchmark.

## Cartesian control and the first server check

`--control-interface eef_pos` is now the default. It uses the documented Inspect YAM 14-D Cartesian interface and the existing agent's `move_to` tool. For each arm the labels are `x, y, z, yaw, pitch, roll, gripper`. Positions are metres in that arm's measured base frame. Yaw/pitch/roll are absolute offsets from the reset tool orientation (`Rz(yaw) @ Ry(-pitch) @ Rx(roll) @ R_reset`); 0 preserves that orientation. Each gripper is independently 0 closed / 1 open.

The simulator backend uses the loaded YAM USD's measured body poses and PhysX Jacobians, including floating-base columns on the world-anchored asset, rather than installing or connecting a CAN hardware driver. The tool point is derived from the loaded distal finger mesh geometry with a 2 mm pad inset. It is not the wrist or the finger-joint origins. Reset writes `control_setup.json` with that offset, its CAD bounds, measured poses, and reference orientations. Unexpected geometry/Jacobians fail before any model call.

Run this first with your own preview stopped:

```bash
cd /home/ubuntu/workspace/robocurve-scenes
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/run_inspect.py --scene chess-mate \
  --mode reach-check --control-interface eef_pos --servo-profile tracking --steps 600 \
  --stream --public-ip 160.211.46.91
```

This makes zero model calls. It first checks that both grippers hold home, then raises each grasp point 15 mm, closes its gripper by 10%, reopens it, and returns it while the other arm holds. It records the measured position/orientation/opening errors in `reach_check.json`, saves all three frame streams, and exports `rollout.mp4`. A phase fails when tracking does not settle within 3 mm, 0.04 rad, and 0.03 normalized opening, or when the task terminates. Errors above 50 mm or 0.3 rad abort early. `reach_trace.jsonl` records the commanded and measured EEF/joint state at each step. Inspect both the report and the video before a paid task attempt.

Cartesian runs default to the run-local `tracking` servo profile: arm stiffness/damping 400/40 and finger stiffness/damping 400/8. Torque limits remain the scene's original limits. The earlier scene profile (80/8 proximal, 20/2 distal, 40/2 fingers for chess) sagged under gravity before a 15 mm reach could be tested. `--servo-profile scene` retains that original profile; `auto` chooses tracking for EEF and scene for raw joints. Original/effective gains are saved in `run.json`; available measured stiffness, damping and drive limits are saved in `control_setup.json`. These controller changes do not rewrite scene YAML or USD files.

The measured `eef_state` drives Cartesian interpolation. Observations also retain joint angles, both world base transforms, world grasp positions, and tool-axis directions. Native Arena action mapping, cameras, scoring and MP4 recording are retained. `last_task_result.json` saves the last task score and hard-failure reasons; terminal reasons are printed too.

For diagnostics only, `--control-interface joints` retains the earlier raw joint interface. Cartesian controls have per-step limits of 2 mm, 0.04 rad and 0.05 opening, and the IK joint command changes are limited to 0.025 rad. The IK correction updates a persistent joint command, preserving the PD offset required to hold against gravity rather than re-seeding from a sagging measured pose each step. These are local servo limits, not arm/table collision planning. Unreachable targets can leave a measurable residual and must not be reported as reached.

## Clearer chess observations for a diagnostic retry

Pass `--camera-resolution 512` to render all three policy sensor views at 512x512.
This updates the composed sensor configuration for that process before cameras
are created; it does not upscale a 256x256 image or modify scene YAML. The
camera pose, lens and field of view remain unchanged. Original/effective sizes
are recorded in `run.json`. The default without this flag preserves scene sizes.
Higher resolution increases camera cost, raw-frame storage and model image input.

The EEF policy docs now supply configured square size, board pose and (for an
identity-oriented board) the file/rank-to-world coordinate formula. This is public
workspace calibration: no piece identities, FEN or mating move are supplied.
It addresses guessed scale and orientation; it does not establish that the model
can reliably recognize pieces or complete a grasp. Review a `--mode capture
--camera-resolution 512` run, then use a short agent attempt with e.g.
`--camera-resolution 512 --max-llm-calls 12` before another 200-call attempt.

## First run: real camera capture, no model calls

Stop your own existing simulation first. Use the existing Arena interpreter:

```bash
cd /home/ubuntu/workspace/robocurve-scenes
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/run_inspect.py --scene chess-mate --mode capture
```

Non-streaming runs explicitly select the installed Isaac Lab headless rendering experience, as the working chess render diagnostic does. A missing or ambiguous rendering experience stops before Kit starts.

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

All three RGB images are sent on each policy decision. The agent has a three-call budget by default and up to three HTTP attempts per model decision. This SDK does not support an output-token cap on its Chat wire, so no per-request token cap is claimed. The exact policy configuration is validated before Kit starts. The limit on environment steps can truncate a motion chunk. The result is a bring-up trial, not the full task benchmark.

Inspect logs, frame sidecars and agent transcripts go into the run's `inspect/` directory. The default upstream policy retains its provider-specific tool metadata between calls.

## Bounded recovery from provider timeouts

`--request-attempts 3` (the default) uses the installed Inspect SDK's native retry
loop for network/read failures, HTTP 429 and HTTP 5xx. It makes up to three total
attempts per model decision, not three additional retries. Exponential backoff
starts at 2 seconds (then 4); a provider Retry-After header takes precedence.
`--retry-backoff-seconds` changes the base delay. Authentication and other
non-transient 4xx errors fail immediately.

The installed Chat client retains its 120-second HTTP timeout settings. There is
no overall wall-clock deadline. Retries resend the same model request while
simulation is paused; they do not restart the environment, repeat a previous
robot move or advance physics. Only a successfully received action chunk is
executed. `--max-llm-calls` limits model decisions, not raw HTTP attempts; 12
decisions can therefore make up to 36 HTTP attempts. Retry settings are recorded
in `run.json`, and exhausting them still preserves the partial MP4 and error log.

## Default MP4 recording for agent runs

Agent and reach-check modes check FFmpeg and the H.264 encoder before prompting for a key or starting the simulator. It uses FFmpeg from PATH, an already-installed `imageio_ffmpeg` binary, or `--ffmpeg /absolute/path/to/ffmpeg`. If none works, the run stops before making model calls.

After a normal end, policy give-up, handled error, or a single Ctrl+C, the launcher exports saved sensor frames before shutting down Kit:

- `<run directory>/rollout.mp4`: overhead, front, side, side by side.
- `<run directory>/videos/overhead.mp4`, `front.mp4`, and `side.mp4`: individual camera views.
- `<run directory>/videos/recording.json`: export status, frame steps, playback rate, and any incomplete camera sets excluded from export.

Playback uses the actual control rate and simulation time. Model-thinking pauses and the desktop streaming viewport are not part of these videos. Raw camera frames, actions, and eval logs remain available. Allow the export to finish after Ctrl+C. A hard kill, second interrupt, or stopped machine can prevent export; saved frames can be recovered separately:

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/robocurve_inspect/video.py \
  --run-dir "/absolute/path/to/the/run" --fps 15
```

Use the run's `control_hz` from `run.json` for `--fps` when it differs from 15. The real media tests encode and decode a combined MP4 and check all three camera colors/order, partial-run recovery without a final JSON log, and missing-view failure without deleting frames.

## Extended run: 120 simulation seconds, 200 model calls

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python \
  integrations/inspect-arena/run_inspect.py --scene chess-mate --mode agent \
  --episode-seconds 120 --steps 2000 --max-llm-calls 200 \
  --stream --public-ip 160.211.46.91
```

This changes only the current environment's timeout before Lab creates it. It leaves the checked-in 90-second task definition unchanged. At 15 Hz the effective step limit is 1800. Success, failure, or policy give-up can end a trial earlier. The 200 calls are an upper bound, not a target to exhaust. There is no wall-clock timeout.

The saved `run.json` records the configured and effective episode durations, whether task timing was modified, the effective step budget, control rate, and model-call cap. Camera frames, actions, Inspect logs, and the default MP4s are saved as usual. A 120-second chess run uses modified task timing; it is not the original 90-second task setting.

## Native joint mapping beneath Cartesian control

The explicit `--control-interface joints` interface has 14 absolute targets:

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

## Consecutive agent moves and stop diagnostics

Agent runs explicitly use Inspect's `ClampApprover` and `DeltaLimitApprover`.
The Python `eval()` API otherwise defaults to an unrestricted `AutoApprover`.
The agent interpolates each new tool motion from measured state, which can differ
slightly from the previous commanded endpoint. The rate limiter bridges that
handoff within the declared command limits; the adapter's own bounds and rate
checks remain enabled. Inspect records modified actions as approval events.

Every returned evaluation writes `eval_summary.json` with status, errors,
completed steps, simulation duration, per-scene stop reasons and scores. An error
log is treated as a failed run even when `eval()` returns normally. Partial MP4
export still runs before Kit closes. A saved MP4 alone does not mean the trial
completed successfully.

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
- https://github.com/robocurve/inspect-robots-yam#cartesian-eef-mode
- https://github.com/isaac-sim/IsaacLab/blob/main/scripts/tutorials/05_controllers/run_diff_ik.py
- https://openrouter.ai/google/gemini-4-argon
- https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/
