"""Per-environment score memory shared by the sim adapter. No Isaac Sim imports."""

from __future__ import annotations

from my_package.scoring.config import BatteryInsertionConfig
from my_package.scoring.scorer import ScoreReport
from my_package.scoring.snapshot import SceneSnapshot

_config: BatteryInsertionConfig | None = None
_snapshots: dict[int, SceneSnapshot] = {}
_reports: dict[int, ScoreReport] = {}
_updated_step: dict[int, int] = {}


def set_active_config(cfg: BatteryInsertionConfig) -> None:
    global _config
    _config = cfg
    clear_all()


def get_active_config() -> BatteryInsertionConfig:
    if _config is None:
        raise RuntimeError("battery insertion config was not registered before the task stepped")
    return _config


def get_snapshot(env_id: int) -> SceneSnapshot | None:
    return _snapshots.get(env_id)


def get_report(env_id: int) -> ScoreReport | None:
    return _reports.get(env_id)


def store(env_id: int, snapshot: SceneSnapshot, report: ScoreReport, step_id: int) -> None:
    _snapshots[env_id] = snapshot
    _reports[env_id] = report
    _updated_step[env_id] = step_id


def already_updated(env_id: int, step_id: int) -> bool:
    return _updated_step.get(env_id) == step_id


def clear_env(env_id: int) -> None:
    _snapshots.pop(env_id, None)
    _reports.pop(env_id, None)
    _updated_step.pop(env_id, None)


def clear_all() -> None:
    _snapshots.clear()
    _reports.clear()
    _updated_step.clear()
