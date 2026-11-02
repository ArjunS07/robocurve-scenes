"""Arena task for one-move checkmate. The description does not name the move."""

from __future__ import annotations

import numpy as np

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.configclass import make_configclass

from my_package.scoring.config import ChessMateConfig
from my_package.tasks.sim_bridge import (
    board_moved,
    mate_state_tracker,
    mate_succeeded,
    multiple_transferred,
    piece_off_table,
    reset_chess_state,
    scores_tensor,
)


class MateScoreRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name

    def record_pre_reset(self, env_ids):
        return self.name, scores_tensor(self._env, env_ids)


@configclass
class MateScoreRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = MateScoreRecorder
    name: str = "mate_score"


def compute_mean_mate_score(recorded_metric_data: list[np.ndarray]) -> float:
    if not recorded_metric_data:
        return 0.0
    values = np.concatenate([np.asarray(item).reshape(-1) for item in recorded_metric_data])
    return float(np.mean(values))


class MateScoreMetric(MetricBase):
    name = "mate_score"
    recorder_term_name = "mate_score"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return MateScoreRecorderCfg(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_mean_mate_score,
            params={},
            recorder_term_name=self.recorder_term_name,
        )


class ChessMateTask(TaskBase):
    def __init__(self, cfg: ChessMateConfig):
        description = (
            "Play White: checkmate Black by moving one white piece onto the mating square and releasing it upright. "
            "The board stays on its mark, with the white pieces nearest the robot. "
            "Every other piece stays on its original square."
        )
        super().__init__(episode_length_s=cfg.scoring.episode_length_s, task_description=description)
        self.cfg = cfg

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="checkmate",
                    predicate_sequence=[mate_succeeded],
                    description="The mating piece is upright, released, and resting inside the mating square.",
                )
            ],
            failures={
                "piece_off_table": TerminationTermCfg(func=piece_off_table),
                "board_moved": TerminationTermCfg(func=board_moved),
                "multiple_transferred": TerminationTermCfg(func=multiple_transferred),
            },
        )

    def get_events_cfg(self):
        return make_configclass(
            "ChessEventsCfg",
            [
                (
                    "reset_joints",
                    EventTermCfg,
                    EventTermCfg(func=reset_all_articulation_joints, mode="reset"),
                ),
                (
                    "reset_pieces",
                    EventTermCfg,
                    EventTermCfg(func=reset_chess_state, mode="reset"),
                ),
            ],
        )()

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("chess mate does not provide a mimic environment")

    def get_metrics(self) -> list[MetricBase]:
        return [MateScoreMetric(), SuccessRateMetric()]

    def get_rewards_cfg(self):
        return make_configclass(
            "ChessRewardsCfg",
            [("mate_state", RewardTermCfg, RewardTermCfg(func=mate_state_tracker, weight=1.0))],
        )()

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.20, -0.85, 0.503), lookat=(0.48, 0.0, 0.033))
