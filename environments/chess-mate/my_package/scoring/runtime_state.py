"""Per-environment score memory. No Isaac Sim imports."""

from __future__ import annotations

from my_package.scoring.config import ChessMateConfig
from my_package.scoring.puzzles import index_for_reset
from my_package.scoring.scorer import ScoreReport
from my_package.scoring.snapshot import SceneSnapshot

_config: ChessMateConfig | None = None
_snapshots: dict[int, SceneSnapshot] = {}
_reports: dict[int, ScoreReport] = {}
_updated_step: dict[int, int] = {}
_reset_ordinal: dict[int, int] = {}
_active_index: dict[int, int] = {}


def set_active_config(cfg: ChessMateConfig) -> None:
    global _config
    _config = cfg
    clear_all()


def get_active_config() -> ChessMateConfig:
    if _config is None:
        raise RuntimeError("chess mate config was not registered before the task stepped")
    return _config


def get_snapshot(env_id: int) -> SceneSnapshot | None:
    return _snapshots.get(env_id)


def get_report(env_id: int) -> ScoreReport | None:
    return _reports.get(env_id)


def active_puzzle_index(env_id: int) -> int:
    return _active_index.get(env_id, 0)


def note_reset(env_id: int, sequence_length: int) -> int:
    """Record a reset and return the puzzle that reset should pose.

    The first reset uses puzzle 0. The next reset advances, wrapping at the end
    of the sequence.
    """
    ordinal = _reset_ordinal.get(env_id, 0)
    index = index_for_reset(ordinal, sequence_length)
    _reset_ordinal[env_id] = ordinal + 1
    _active_index[env_id] = index
    _snapshots.pop(env_id, None)
    _reports.pop(env_id, None)
    _updated_step.pop(env_id, None)
    return index


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
    _reset_ordinal.clear()
    _active_index.clear()
