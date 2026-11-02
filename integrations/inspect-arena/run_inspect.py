"""Own Inspect/Arena bridge. Default mode captures cameras with no model calls."""
import argparse
from datetime import datetime, timezone
import getpass
import subprocess
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4


def create_agent_policy(model, max_llm_calls, *, transport=None, env=None, request_attempts=3, backoff_s=2.,
                        effort='high', cache_friendly=True, read_timeout_s=300.):
    """One tested configuration for the installed Inspect agent's Chat wire."""
    from inspect_robots_agent.policy import LLMAgentPolicy
    from robocurve_inspect.openrouter import OpenRouterTransport
    telemetry = OpenRouterTransport(transport, cache_friendly=cache_friendly, read_timeout_s=read_timeout_s)
    policy = LLMAgentPolicy(model=model, base_url='https://openrouter.ai/api/v1',
                          api_key_env='OPENROUTER_API_KEY', wire='chat', images='always',
                          depth='off', wire_capture=False, effort=effort, image_horizon=None if cache_friendly else 2, max_retries=request_attempts, backoff_s=backoff_s,
                          max_llm_calls=max_llm_calls, transcript_echo=True,
                          transport=telemetry, env=env)
    policy.robocurve_transport = telemetry
    return policy


def create_motion_approver(action_space):
    """Use Inspect's recorded rate limiter across measured-state replanning boundaries."""
    from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
    return ChainApprover(ClampApprover(action_space), DeltaLimitApprover(action_space))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('chess-mate', 'battery-remote', 'atb1-nut'), required=True)
    parser.add_argument('--mode', choices=('capture', 'hold', 'agent', 'reach-check'), default='capture')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--control-interface', choices=('eef_pos','joints'), default='eef_pos')
    parser.add_argument('--servo-profile',choices=('auto','scene','tracking'),default='auto')
    parser.add_argument('--episode-seconds', type=float,
                        help='Override simulation-time timeout for this run; recorded as modified task timing')
    parser.add_argument('--arena-root', type=Path, default=Path('/home/ubuntu/workspace/IsaacLab-Arena'))
    parser.add_argument('--scene-config', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--camera-resolution', type=int, choices=(256,512,768,1024),
                        help='Run-local square RGB resolution for all three policy cameras')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--public-ip', default=os.environ.get('PUBLIC_IP'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ffmpeg', help='Optional FFmpeg executable for default agent MP4 recording')
    parser.add_argument('--model', default='google/gemini-4-argon')
    parser.add_argument('--reasoning-effort', choices=('low','medium','high'), default='high')
    parser.add_argument('--cache-friendly', action=argparse.BooleanOptionalAction, default=True,
                        help='Stable provider session and full image history for implicit prompt caching')
    parser.add_argument('--request-read-timeout', type=float, default=300.)
    parser.add_argument('--max-llm-calls', type=int, default=3)
    parser.add_argument('--request-attempts', type=int, default=3,
                        help='Maximum HTTP attempts per model decision, including the initial request')
    parser.add_argument('--retry-backoff-seconds', type=float, default=2.)
    args = parser.parse_args()
    if args.steps <= 0 or args.max_llm_calls <= 0:
        parser.error('--steps and --max-llm-calls must be positive')
    if not math.isfinite(args.request_read_timeout) or args.request_read_timeout <= 0:
        parser.error('--request-read-timeout must be positive and finite')
    if args.request_attempts <= 0:
        parser.error('--request-attempts must be positive')
    if not math.isfinite(args.retry_backoff_seconds) or args.retry_backoff_seconds < 0:
        parser.error('--retry-backoff-seconds must be finite and nonnegative')
    if args.episode_seconds is not None and (not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0):
        parser.error('--episode-seconds must be positive and finite')
    if args.mode=='reach-check' and args.control_interface!='eef_pos':
        parser.error('reach-check requires --control-interface eef_pos')
    if args.stream and not args.public_ip:
        parser.error('--stream needs --public-ip or PUBLIC_IP')
    try:
        import inspect_robots
        if args.mode == 'agent':
            from inspect_robots_agent.policy import LLMAgentPolicy
    except ImportError as exc:
        parser.error(f'Use the existing Arena interpreter with Inspect installed: {exc}')
    # Camera recording is mandatory for agent/reach-check runs: reject an unusable encoder
    # before prompting for a key, creating Kit, or making any model request.
    from robocurve_inspect.video import require_ffmpeg, export_recording
    encoder = None
    if args.mode in ('agent','reach-check'):
        try:
            encoder = require_ffmpeg(args.ffmpeg)
        except RuntimeError as exc:
            parser.error(str(exc))
    if args.mode == 'agent' and not os.environ.get('OPENROUTER_API_KEY'):
        if not sys.stdin.isatty():
            parser.error('Set OPENROUTER_API_KEY in the process environment or run in an interactive terminal')
        os.environ['OPENROUTER_API_KEY'] = getpass.getpass('OpenRouter API key (hidden, not saved): ').strip()
        if not os.environ['OPENROUTER_API_KEY']:
            parser.error('An OpenRouter key is required for agent mode')
    # Validate the exact provider settings before spending time starting Kit.
    # This constructor does not make a completion request.
    policy = create_agent_policy(args.model, args.max_llm_calls, request_attempts=args.request_attempts,
                                 backoff_s=args.retry_backoff_seconds, effort=args.reasoning_effort,
                                 cache_friendly=args.cache_friendly, read_timeout_s=args.request_read_timeout) if args.mode == 'agent' else None
    # Import torch before Arena. Other adapter modules are numpy/Inspect-only.
    import torch
    from inspect_robots import Action, Scene, Task, eval, success_at_end
    from robocurve_inspect.adapter import ArenaTaskScore
    from robocurve_inspect.artifacts import save_capture
    from robocurve_inspect.runtime import open_scene
    output = args.output or (Path(__file__).resolve().parent / 'outputs' /
              f'{args.scene}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if policy is not None:
        policy.robocurve_transport.output = output
    provenance = {'scene': args.scene, 'mode': args.mode, 'python': sys.executable,
                  'control_interface':args.control_interface,
                  'model': args.model if args.mode == 'agent' else None,
                  'reasoning_effort': args.reasoning_effort if args.mode == 'agent' else None,
                  'prompt_caching': {'strategy':'gemini_implicit', 'cache_friendly':args.cache_friendly,
                                     'image_horizon':None if args.cache_friendly else 2,
                                     'session_id':policy.robocurve_transport.session_id if policy is not None else None,
                                     'cache_hit_guaranteed':False},
                  'request_read_timeout_seconds':args.request_read_timeout,
                  'endpoint': 'https://openrouter.ai/api/v1' if args.mode == 'agent' else None,
                  'task_clock': 'simulation time; model inference does not advance physics',
                  'requested_episode_seconds': args.episode_seconds,
                  'requested_max_environment_steps': args.steps,
                  'max_llm_calls': args.max_llm_calls if args.mode == 'agent' else None,
                  'request_attempts_per_decision': args.request_attempts if args.mode == 'agent' else None,
                  'retry_backoff_seconds': args.retry_backoff_seconds if args.mode == 'agent' else None,
                  'motion_guardrails': 'Inspect ClampApprover + DeltaLimitApprover; adapter bounds/rate checks retained',
                  'video': {'enabled': args.mode in ('agent','reach-check'), 'encoder': encoder,
                            'composite': 'rollout.mp4' if args.mode in ('agent','reach-check') else None},
                  'versions': {}}
    repo = Path(__file__).resolve().parents[2]
    try:
        provenance['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        provenance['git_changes'] = subprocess.check_output(
            ['git', 'status', '--porcelain', '--', 'integrations/inspect-arena', 'environments/' + args.scene],
            cwd=repo, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        provenance['git_commit'] = None
    for name in ('inspect-robots', 'inspect-robots-agent', 'isaaclab', 'isaacsim'):
        try:
            provenance['versions'][name] = version(name)
        except PackageNotFoundError:
            provenance['versions'][name] = None
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Outputs: {output}', flush=True)
    with open_scene(args, output) as (adapter, instruction):
        provenance.update(adapter.base._robocurve_inspect_episode_budget)
        provenance['control_hz'] = adapter.info.control_hz
        provenance['servo_settings'] = adapter.base._robocurve_inspect_servo_report
        provenance['camera_settings'] = adapter.base._robocurve_inspect_camera_report
        max_episode = max(1, math.ceil(float(adapter.base.cfg.episode_length_s) * adapter.info.control_hz))
        provenance['effective_max_environment_steps'] = min(args.steps, max_episode)
        (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f"Episode budget: {provenance['effective_episode_seconds']:g} simulation seconds; "
              f"up to {provenance['effective_max_environment_steps']} steps; "
              f"model-call limit: {provenance['max_llm_calls']}", flush=True)
        if args.mode == 'agent':
            print(f'Model requests: up to {args.request_attempts} attempts per decision; '
                  f'exponential retry backoff starts at {args.retry_backoff_seconds:g}s', flush=True)
            print(f'Reasoning: {args.reasoning_effort}; cache-friendly routing: {args.cache_friendly}; '
                  f'read timeout: {args.request_read_timeout:g}s', flush=True)
        scene = Scene(id=args.scene, instruction=instruction)
        if args.mode == 'reach-check':
            from robocurve_inspect.probe import run_reach_check
            probe_failed=False
            try:
                report=run_reach_check(adapter,scene,output,provenance['effective_max_environment_steps'])
                print(f"Reach check passed: {report['passed']}; report: {output/'reach_check.json'}",flush=True)
            except BaseException:
                probe_failed=True
                raise
            finally:
                try:
                    export_recording(output,encoder,adapter.info.control_hz)
                except Exception as exc:
                    print(str(exc),file=sys.stderr,flush=True)
                    if not probe_failed: raise
        elif args.mode == 'agent':
            task = Task(name=args.scene, scenes=[scene], scorer=[ArenaTaskScore(), success_at_end()],
                        max_steps=min(args.steps, max_episode))
            rollout_failed = False
            try:
                logs = eval(task, policy, adapter, log_dir=str(output / 'inspect'), store_frames=True,
                            fail_on_error=True, environment_revision=provenance['git_commit'],
                            approver=create_motion_approver(adapter.info.action_space))
                from robocurve_inspect.results import save_eval_summary
                summary = save_eval_summary(logs, output, adapter.info.control_hz)
                for run in summary['runs']:
                    print(f"Inspect status: {run['status']}; {run['total_steps']} steps / "
                          f"{run['simulation_seconds']:.2f} simulation seconds; "
                          f"stop reasons: {[s['termination_reasons'] for s in run['scenes']]}", flush=True)
                if summary['errors']:
                    raise RuntimeError('Inspect rollout failed: ' + '; '.join(summary['errors']))
                print(f'Inspect log directory: {output / "inspect"}', flush=True)
            except BaseException:
                rollout_failed = True
                raise
            finally:
                # Export before leaving the Kit context: its exception handler
                # can terminate the process. Do not mask the rollout's error.
                if adapter.last_result is not None:
                    (output/'last_task_result.json').write_text(json.dumps(adapter.last_result,indent=2)+'\n')
                print('Exporting camera MP4s from saved frames...', flush=True)
                try:
                    export_recording(output, encoder, adapter.info.control_hz)
                except Exception as exc:
                    print(str(exc), file=sys.stderr, flush=True)
                    if not rollout_failed:
                        raise
        else:
            observation = adapter.reset(scene, seed=42)
            save_capture(output / 'initial', observation, adapter.info, adapter._report())
            count = min(args.steps, 2) if args.mode == 'capture' else args.steps
            # Hold the measured initial state; never issue an arbitrary zero absolute target.
            target = observation.state[adapter.command_state_key].copy()
            for index in range(count):
                result = adapter.step(Action(data=target))
                observation = result.observation
                if result.terminated or result.truncated:
                    print(f'Episode ended: {result.termination_reason}; task score={result.info["task_score"]}')
                    break
            save_capture(output / 'latest', observation, adapter.info, adapter._report() if not adapter._done else result.info['task_report'])
            print(f'Camera check complete: {output / "latest" / "cameras.html"}')
            print('Camera order: overhead | front | side. No model calls made.')


if __name__ == '__main__':
    main()
