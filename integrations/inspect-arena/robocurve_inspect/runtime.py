"""Arena launch boundary. All Isaac imports happen after torch and Kit startup."""
from contextlib import contextmanager
from importlib import import_module
import math
import os
from pathlib import Path
import sys
from .adapter import ArenaEmbodiment
from .contract import scene_arms

SCENES = {
    'chess-mate': ('chess_environment:ChessMateEnvironment', 'chess_mate', 'chess_config'),
    'battery-remote': ('battery_environment:BatteryInsertionEnvironment', 'battery_insertion', 'battery_config'),
    'atb1-nut': ('nut_environment:NutThreadingEnvironment', 'nut_threading', 'nut_config'),
}


def rendering_args(arena_root, stream):
    """Enable RTX camera rendering using this installed Lab's experience file."""
    if stream:
        return ['--livestream', '1']
    apps = Path(arena_root) / 'submodules' / 'IsaacLab' / 'apps'
    candidates = sorted(path for path in apps.glob('*.kit')
                        if 'headless' in path.name and 'rendering' in path.name)
    preferred = apps / 'isaaclab.python.headless.rendering.kit'
    if preferred.is_file():
        experience = preferred
    elif len(candidates) == 1:
        experience = candidates[0]
    else:
        raise RuntimeError(f'Cannot select a headless camera-rendering experience in {apps}; '
                           f'found {[path.name for path in candidates]}')
    return ['--experience', str(experience.resolve())]


def configure_episode_limit(env_cfg, seconds):
    """Override only this run's Lab timeout; leave the scene YAML untouched."""
    configured = float(env_cfg.episode_length_s)
    effective = configured if seconds is None else float(seconds)
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError('Episode duration must be positive and finite')
    if seconds is not None:
        # Arena's time_out term uses Lab's max_episode_length derived from this
        # setting. Its scene task config remains the original task definition.
        env_cfg.episode_length_s = effective
    return {'configured_episode_seconds': configured,
            'effective_episode_seconds': effective,
            'task_timing_modified': effective != configured}


def install_terminal_recorder(env_cfg):
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg, RecorderManagerBaseCfg
    from isaaclab.utils import configclass
    from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

    class TerminalCapture(RecorderTerm):
        def record_pre_reset(self, env_ids):
            callback = getattr(self._env, '_robocurve_inspect_capture', None)
            # Only one environment is supported. Empty lists do not denote a reset.
            if callback is not None and env_ids is not None and len(env_ids):
                callback()
            return None, None

    @configclass
    class TerminalCaptureCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = TerminalCapture

    extra = make_configclass('InspectCaptureCfg', [('inspect_terminal', RecorderTermCfg, TerminalCaptureCfg())])()
    env_cfg.recorders = combine_configclass_instances('InspectRecorderCfg', env_cfg.recorders, extra,
                                                      bases=(RecorderManagerBaseCfg,))


def report_reader(scene):
    if scene == 'atb1-nut':
        bridge = import_module('my_package.tasks.sim_bridge')
        def read(env):
            # cfg is attached by the launch context, not guessed from privileged observations.
            bridge.advance_all(env, env._robocurve_inspect_task_cfg)
            return bridge.runtimes_for(env)[0].last_report
        return read
    bridge = import_module('my_package.tasks.sim_bridge')
    state = import_module('my_package.scoring.runtime_state')
    def read(env):
        bridge.update_from_env(env)
        return state.get_report(0)
    return read


@contextmanager
def open_scene(args, output):
    repo = Path(__file__).resolve().parents[3]
    scene_root = repo / 'environments' / args.scene
    if not scene_root.is_dir():
        raise FileNotFoundError(f'Scene is not present in this checkout: {scene_root}')
    if not (args.arena_root / 'isaaclab_arena').is_dir():
        raise FileNotFoundError(f'Arena installation not found: {args.arena_root}')
    # Distinct scenes all use the package name my_package; isolate them by process.
    if 'my_package' in sys.modules:
        raise RuntimeError('Launch one scene per fresh process; my_package is already imported')
    sys.path.insert(0, str(args.arena_root))
    sys.path.insert(0, str(scene_root))
    os.environ['ROBOCURVE_GENERATED_DIR'] = str(output / 'generated')
    os.environ.setdefault('YAM_USD', '/home/ubuntu/workspace/robocurve/assets/yam/yam_source_resolved/yam_source_resolved.usda')
    if args.stream:
        if not args.public_ip:
            raise ValueError('--stream needs --public-ip or PUBLIC_IP')
        os.environ['PUBLIC_IP'] = args.public_ip
    import torch  # must precede importing Arena/Kit
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    factory, name, config_flag = SCENES[args.scene]
    native_args = ['--visualizer', 'kit' if args.stream else 'none', '--device', args.device,
                   '--num_envs', '1', '--enable_cameras', '--no_solve_relations',
                   '--external_environment_class_path', 'my_package.isaaclab_arena_environments.' + factory]
    native_args += rendering_args(args.arena_root, args.stream)
    native_args += [name]
    if args.scene_config:
        native_args += ['--' + config_flag, str(args.scene_config.resolve())]
    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *native_args]
    adapter = env = None
    try:
        parser = get_isaaclab_arena_cli_parser()
        initial, _ = parser.parse_known_args(native_args)
        with SimulationAppContext(initial):
            from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
            parser = get_isaaclab_arena_environments_cli_parser(parser)
            native = parser.parse_args(native_args)
            if args.scene == 'atb1-nut':
                # This older scene has a module default instead of honoring the output env var.
                geometry = import_module('my_package.assets.geometry')
                geometry.GENERATED_DIR = output / 'generated'
            builder = get_arena_builder_from_cli(native)
            env_cfg, env_kwargs = builder.compose_manager_cfg()
            episode_budget = configure_episode_limit(env_cfg, getattr(args, 'episode_seconds', None))
            install_terminal_recorder(env_cfg)
            env_cfg.recorders.dataset_export_dir_path = str(output / 'arena')
            try:
                env = builder.make_registered(env_cfg=env_cfg, env_kwargs=env_kwargs)
                base = env.unwrapped
                base._robocurve_inspect_episode_budget = episode_budget
                task = builder.arena_env.task
                cfg = task.cfg
                base._robocurve_inspect_task_cfg = cfg
                if args.scene == 'atb1-nut':
                    sensors = ('overhead', 'front', 'side')
                else:
                    sensors = ('overhead_cam', 'front_cam', 'side_cam')
                cameras = {}
                for label, sensor_name in zip(('overhead', 'front', 'side'), sensors):
                    sensor = base.scene[sensor_name]
                    cameras[label] = (sensor_name, int(sensor.cfg.height), int(sensor.cfg.width))
                arms=scene_arms(args.scene,cfg)
                if getattr(args,'control_interface','eef_pos')=='eef_pos':
                    from .cartesian import CartesianEmbodiment
                    rig_docs=''
                    table=getattr(cfg,'table',None)
                    if table is not None:
                        rig_docs=f'Table surface world Z={table.top_z_m} m. '
                    board=getattr(cfg,'board_pose',None)
                    if board is not None:
                        rig_docs+=f'Board center world XYZ={tuple(board.position_xyz)} m. '
                        rig_docs+='White home rank is toward world -X; files a to h run toward world -Y. '
                    adapter=CartesianEmbodiment(env,arms,cameras,report_reader(args.scene),rig_docs=rig_docs)
                    adapter.control_report_path=output/'control_setup.json'
                else:
                    adapter=ArenaEmbodiment(env,arms,cameras,report_reader(args.scene))
                controller = getattr(base, 'viewport_camera_controller', None)
                if controller is not None:
                    controller.update_view_location()
                yield adapter, base.get_language_instruction()
            finally:
                if adapter is not None:
                    adapter.close()
                if env is not None:
                    env.close()
    finally:
        sys.argv = previous_argv
