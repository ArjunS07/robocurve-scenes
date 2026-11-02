# robocurve-scenes

Three bimanual manipulation tasks for [Isaac Lab-Arena](https://isaac-sim.github.io/IsaacLab-Arena/), each with its own scene, scoring, launcher, and checks. All three use two YAM arms on the same maple table.

| Task | Directory | Arena environment | Episode | Score metric |
| --- | --- | --- | --- | --- |
| Chess: mate in one (Task 7) | `environments/chess-mate` | `chess_mate` | 90 s | `mate_score` |
| TV-remote batteries (Task 6) | `environments/battery-remote` | `battery_insertion` | 90 s | `insertion_score` |
| Nut threading (ATB1-inspired) | `environments/atb1-nut` | `nut_threading` | 120 s | `nut_score` |

Each task directory is a self-contained package (`my_package/`) with:

- `my_package/isaaclab_arena_environments/` – the Arena environment class
- `my_package/assets/` – geometry/USD generation and spawn configs
- `my_package/scoring/` – the scorer (pure Python, no Isaac imports) and its config loader
- `my_package/tasks/` – the Arena task (terminations, metrics, events) and the bridge that reads simulator state into the scorer
- `my_package/embodiments/` – the YAM bimanual embodiment (14-dim action)
- `my_package/config/` or `config/` – the task's YAML parameters
- `launch_*.py` – the launcher
- `tests/` – pure tests (no simulator)
- `tools/` – server-side physics/render checks
- `STATUS.md` – what is verified on the server, what is only code-checked, and what is still open
- `INTEGRATION.md` – server paths and troubleshooting

`environments/tools/probe_runtime.py` is a read-only probe of the installed runtime (quaternion order, table geometry, YAM joints and bodies).

## Setup

### Server (where the simulator runs)

The scenes run against an existing native Isaac Lab-Arena installation. Nothing is installed by this repository.

| Requirement | Value used here |
| --- | --- |
| Arena checkout | `/home/ubuntu/workspace/IsaacLab-Arena` (Isaac Lab 17.0.2 / Kit 110.3 submodule) |
| Interpreter | `/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python` |
| YAM robot USDA | `/home/ubuntu/workspace/robocurve/assets/yam/yam_source_resolved/yam_source_resolved.usda` |
| GPU | NVIDIA L40S (any RTX GPU with enough memory should work) |

```bash
cd /home/ubuntu/workspace
git clone https://github.com/ArjunS07/robocurve-scenes.git
cd robocurve-scenes
export PY=/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python
```

Do not run `uv sync`, reinstall dependencies, or edit the Arena checkout, its lockfile, or the YAM asset. Each launcher:

- puts its task directory and the Arena checkout on `sys.path`;
- imports `torch` before Kit starts (required on this machine to avoid a CUDA library conflict);
- writes generated USD to a fresh `outputs/assets-*` directory (`ROBOCURVE_GENERATED_DIR`), never into the source tree;
- writes runner reports to the task's `outputs/` directory (git-ignored).

If your Arena checkout or robot asset lives elsewhere, pass `--arena-root PATH` to the launchers and change `robot.usda_path` (chess, battery) or `robot.usd_path` (nut) in the task YAML.

**Table asset.** Arena 0.3 points `maple_table_robolab` at NVIDIA's *staging* S3 bucket, where the Isaac 6.1 file now returns 404. Each scene loads the identical file from the *production* bucket instead (`my_package/assets/table.py`). Set `ROBOCURVE_MAPLE_TABLE_USD=/path/to/maple_table.usda` to use a local copy.

### Local (pure tests only)

The scorers, geometry generators and configs have no Isaac imports and are tested with plain Python 3.11+ and PyYAML:

```bash
cd environments/chess-mate      && python3 tests/run_pure_tests.py
cd ../battery-remote             && python3 tests/run_pure_tests.py
cd ../atb1-nut                   && python3 tests/run_pure_tests.py
```

Chess also needs numpy and Pillow to rebuild its committed assets (`tools/build_chess_assets.py`); launching never needs them.

## Running a task

All launchers accept the same core options:

| Option | Effect |
| --- | --- |
| *(none)* | 50 zero-action steps, no viewport: a startup and physics smoke test |
| `--num-steps N` | run N environment steps (the environment steps at 15 Hz) |
| `--stream --public-ip IP` | Kit viewport for the Omniverse WebRTC client; defaults to 1,000,000 steps |
| `--enable-cameras` | create the three task cameras (`overhead`, `front`, `side`) |
| `--arena-root PATH` | a different Arena checkout |

Battery and nut also take `--capture-dir DIR` (with `--enable-cameras`), which saves the task cameras' RGB at the steps in `ROBOCURVE_CAPTURE_STEPS` (default `1,25,49`). Battery takes `--single-slot`; nut takes `--nut-config FILE`.

```bash
# Chess
cd environments/chess-mate && $PY launch_chess_mate.py

# Battery
cd environments/battery-remote && $PY launch_battery_insertion.py

# Nut
cd environments/atb1-nut && $PY launch_nut_threading.py
```

The launchers run Arena's `policy_runner.py` with the `zero_action` policy: the arms hold their home pose and nothing in the scene is manipulated. Use them to check startup, physics, and the scorer wiring, not task success.

### A full scored episode

The default 50 steps end before any episode does, so the runner prints `num_episodes: 0`. To score a full episode, run past the episode limit (90 s × 15 Hz = 1350 steps for chess and battery, 120 s × 15 Hz = 1800 steps for the nut):

```bash
cd environments/chess-mate     && $PY launch_chess_mate.py --num-steps 1400
cd environments/battery-remote && $PY launch_battery_insertion.py --num-steps 1400
cd environments/atb1-nut       && $PY launch_nut_threading.py --num-steps 1900
```

The episode ends on timeout (or earlier on success or a hard failure). The score is recorded just before Arena resets the environment and appears in the final line, e.g. `Metrics: {'num_episodes': 1, 'mate_score': 0.0, 'success_rate': 0.0}`, and in `outputs/<timestamp>/index.html`. With the zero-action policy every task scores 0 (nothing is lifted), which is the correct score for doing nothing.

To drive a task with another policy, call the runner directly with the same flags and a different `--policy_type`. Global options must come before the environment name:

```bash
$PY /home/ubuntu/workspace/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \
  --visualizer none --device cuda:0 --num_envs 1 --policy_type <policy> --num_steps 1400 \
  --no_solve_relations \
  --external_environment_class_path my_package.isaaclab_arena_environments.chess_environment:ChessMateEnvironment \
  chess_mate
```

(run from the task directory with that directory on `PYTHONPATH`, and `import torch` before Kit as the launchers do). The action is 14 numbers: for each arm, six joint offsets from the home pose and one gripper command (an opening offset in chess and battery; in the nut scene a positive value closes the gripper). See each task's `my_package/scoring/action_map.py` (chess, battery) or `my_package/embodiments/action_map.py` (nut).

The Inspect Robots integration (`integrations/inspect-arena`) is maintained separately; see its README.

### Verification tools (server, no Arena runner)

These build the same scene in a standalone Kit app, run scripted trials, print one JSON report, and save 1280×720 renders (including collision overlays) under the task's `outputs/`.

| Task | Command | What it checks |
| --- | --- | --- |
| Chess | `$PY tools/render_check.py` | pieces settle without drift, tilt or penetration; robot base vs. tabletop; board, piece and collider renders |
| Battery | `$PY tools/physics_check.py` | fixture holds; springs compress and return; insertion and retention in both slots; reversed polarity rejected; lateral kick; pull-out |
| Nut | `$PY tools/physics_check.py` | engage on the thread; advance per turn; release mid-thread and retention; pull-off; seat; back-off |

Add `--no-render` for physics only. The scripts run with Fabric off by default so renders match the physics; `--use-fabric` reproduces the environment's setting (output in a separate `<revision>-fabric` folder). The Isaac Lab 17 `AppLauncher` has no `--enable_cameras`, so the scripts select the headless *rendering* Kit experience themselves.

## Scoring

Each scorer (`my_package/scoring/scorer.py`) runs once per control step from a snapshot of the simulator state. It keeps per-episode memory (ever lifted, latched failures, retention timers) that is cleared on reset. Failures latch for the episode and override every credit tier; the off-table and fixture or board-moved failures also terminate the episode. The score reported for an episode is the score at its last step, recorded before Arena's reset.

Some quantities are **proxies**, and the code and docs say so:

- **Holding / release** (all tasks) is gripper-body distance plus finger closure, not a contact test. If the gripper body or finger joints cannot be found, holding is *unknown* and the object is never counted as released.
- **Damage** (battery) is a 30 N contact-force threshold, not visible damage.

### Chess: mate in one (`mate_score`)

White to move, white pieces nearest the robot. The puzzle (`fen_sequence` in `my_package/config/chess_mate.yaml`) has exactly one mating move, a single-piece quiet move (here the white queen d1→h5). The solver in `my_package/scoring/chess_rules.py` derives it, and the loader rejects puzzles that are Black to move or whose mate is a capture, castle, promotion, or en passant. The answer is written only to the evaluator sheet (`my_package/assets/generated/puzzle_record.txt`), never to the task string.

| Score | Condition |
| --- | --- |
| 100 | the correct piece, after having been lifted, is released upright and resting with its base entirely inside the mating square, and every other piece is on its original square |
| 75 | the correct piece is upright on the mating square but still held |
| 50 | the correct piece was lifted clear of the board |
| 25 | a different piece was lifted and the correct piece never was |
| 0 | nothing lifted, or a hard failure: a piece off the table, the board moved off its mark (5 mm / 5°), or two or more pieces transferred to new squares |

Sliding the mating piece without lifting it earns no 100 or 75.

### Battery insertion (`insertion_score`)

Two AAA cells lie on marked mats beside a secured, face-down remote with its cover removed. The two cells are interchangeable: either may go in either slot, and polarity is judged against the slot it is in.

A cell counts as **inserted** only when, for 1 s continuously, it is:

- fully seated (within ±1.5 mm of the seat along the slot and 2 mm radially);
- correctly oriented (within 20° of the slot's polarity);
- touching that slot's spring plunger;
- released.

| Score | Condition |
| --- | --- |
| 100 | both cells inserted |
| 50 | one cell inserted |
| 25 | a cell was lifted clear of the table but none is inserted |
| 0 | nothing lifted, or a hard failure: a cell off the table, the remote moved off its mark (5 mm / 5°), or the damage proxy exceeded |

A reversed or partly seated cell does not count. If a slot's spring joint cannot be read, that slot is *unknown* and earns no insertion credit.

### Nut threading (`nut_score`)

One approximate M8 × 1.25 nut-and-stud station inspired by NIST Assembly Task Board 1 (not the full board or its CAD). The stud carries a physical helical ridge and the nut three pins, so turning the nut drives it down the stud through ordinary contacts.

The scorer checks each step that rotation and descent stay consistent with the 1.25 mm pitch while the nut is coaxial. If they ever disagree beyond tolerance, the nut is *cross-threaded* until it is backed off or removed.

| Score | Condition |
| --- | --- |
| 100 | threaded, seated (within 1 mm of the seat), and released |
| 75 | threaded through at least 75% of the travel (held or released) |
| 50 | threaded and released short of 75% |
| 25 | lifted but not threaded, or threaded short of 75% and still held |
| 0 | never lifted, or a hard failure: the nut off the table or the fixture moved |

## What has been demonstrated

From each task's `STATUS.md`, which has the details and the code revision each result was taken at:

- **All three** start under the Arena runner and complete 50 zero-action steps with no false failure.
- **Chess:**
  - the pieces and board settle within 0.6 mm and 0.02° on the new colliders;
  - the board renders without z-fighting;
  - the arms have collision geometry.
- **Battery:** in a scripted physics check (cells placed by pose, no robot):
  - the remote fixture holds;
  - the springs compress and return;
  - a cell seats and stays seated after release in either slot;
  - a reversed cell is rejected;
  - cells stay on their marks at the start.
- **Nut:** in a scripted check (torque applied directly, no robot):
  - the nut engages the thread;
  - it advances about 1.15 mm per turn against a 1.25 mm pitch;
  - it stays in place when released mid-thread.

Not yet demonstrated for any task:

- a full scored episode through the runner;
- a robot actually grasping and moving a piece, cell, or nut;
- policy camera images that track moving bodies (render checks showed stale poses with Fabric on; under investigation).

The nut's continuous advance over several turns is being fixed; it stalled at about 0.4 turns in the last run.
