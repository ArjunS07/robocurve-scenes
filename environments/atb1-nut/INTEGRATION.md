# Nut threading: running it

One approximate M8 nut-and-stud station inspired by NIST Assembly Task Board 1 (not the full board, not its CAD). See STATUS.md for what is verified.

Server layout: repository `/home/ubuntu/workspace/robocurve-scenes`, Arena `/home/ubuntu/workspace/IsaacLab-Arena`, interpreter `/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python`. The YAM USDA path in `config/nut_threading.yaml` is absolute; `YAM_USD` overrides it. This package does not modify the YAM asset, `uv.lock`, or IsaacLab-Arena.

## Pure tests (no simulator)

```bash
cd environments/atb1-nut
python3 tests/run_pure_tests.py
```

## Thread physics check (server, no robot, no Arena runner)

```bash
cd /home/ubuntu/workspace/robocurve-scenes/environments/atb1-nut
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/physics_check.py            # with renders
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/physics_check.py --no-render
```

It prints `ROBOCURVE_PHYSICS_SETUP` before Kit starts and one JSON block between `ROBOCURVE_PHYSICS_CHECK_BEGIN` and `_END`, and writes `report.json` and PNGs to `outputs/physics_check/<revision>/`. Pass criteria: `advance.descent_per_turn_mm` ≈ 1.25, small `max_helix_deviation_mm` and `max_tilt_deg`, `release.retained` true with score 50, `pull.held_on_stud` true, `seat.score_released_seated` 100, `back_off.freed` true, `settle.fixture_drift_mm` 0.

## Scene smoke run and preview

```bash
cd /home/ubuntu/workspace/robocurve-scenes/environments/atb1-nut
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_nut_threading.py                       # 50 zero-action steps
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_nut_threading.py --enable-cameras --capture-dir outputs/captures
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_nut_threading.py --stream --public-ip 160.211.46.91
```

The launcher imports torch first, then runs Arena's `policy_runner.py` with `--visualizer none` (or `kit --livestream 1` when streaming), `--device cuda:0`, one env, `zero_action`, `--num_steps` (50, or 1000000 when streaming), `--output_base_dir outputs`, `--no_solve_relations`, the external environment class, then the `nut_threading` subcommand (and `--nut_config` if given). Generated USDA goes to a fresh `outputs/assets-*` via `ROBOCURVE_GENERATED_DIR`. `--capture-dir` saves the task cameras at steps `ROBOCURVE_CAPTURE_STEPS` (default `1,25,49`). Stream with the WebRTC client at `global.prd.ga.run.brev.nvidia.com`, signal 24317, stream 28337; one streaming scene at a time.

## If startup fails

- `nut threading: gripper body ... not found` lists the live names; set `robot.gripper_body_name`. Until then the nut never counts as released.
- A quaternion problem shows as an immediate `fixture_moved`; check `SIM_QUAT_ORDER` in `my_package/tasks/state_values.py` (measured xyzw).
