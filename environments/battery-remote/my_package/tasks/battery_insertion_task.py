"""Arena task: retention success, hard failures, and a deterministic score reset."""

from __future__ import annotations

import numpy as np

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.configclass import make_configclass

from my_package.scoring.config import BatteryInsertionConfig, active_slots
from my_package.tasks.sim_bridge import (
    batteries_retained,
    battery_off_table,
    damage_proxy,
    insertion_state_tracker,
    remote_moved,
    reset_battery_insertion_state,
    scores_tensor,
)


class InsertionScoreRecorder(RecorderTerm):
    """Record the integer task score before a reset. 100 is the only full success."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name

    def record_pre_reset(self, env_ids):
        return self.name, scores_tensor(self._env, env_ids)


@configclass
class InsertionScoreRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = InsertionScoreRecorder
    name: str = "insertion_score"


def compute_mean_insertion_score(recorded_metric_data: list[np.ndarray]) -> float:
    if not recorded_metric_data:
        return 0.0
    values = np.concatenate([np.asarray(item).reshape(-1) for item in recorded_metric_data])
    return float(np.mean(values))


class InsertionScoreMetric(MetricBase):
    name = "insertion_score"
    recorder_term_name = "insertion_score"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return InsertionScoreRecorderCfg(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_mean_insertion_score,
            params={},
            recorder_term_name=self.recorder_term_name,
        )


class BatteryInsertionTask(TaskBase):
    def __init__(self, cfg: BatteryInsertionConfig):
        if cfg.single_slot:
            description = (
                "Insert one AAA battery into slot A of the face-down remote. "
                "The cover is already off. The positive terminal must match the red mark."
            )
        else:
            description = (
                "Insert both AAA batteries into the face-down remote. The cover is already off. "
                "Slot A keeps the starting polarity and slot B requires the opposite direction."
            )
        super().__init__(episode_length_s=cfg.scoring.episode_length_s, task_description=description)
        self.cfg = cfg

    def get_scene_cfg(self):
        if not self.cfg.scoring.enable_contact_sensors:
            return None
        fields = []
        for slot in active_slots(self.cfg):
            plunger = "{ENV_REGEX_NS}/remote/" + slot.name.replace("battery_", "slot_") + "_plunger_link"
            fields.append(
                (
                    f"contact_{slot.name}",
                    ContactSensorCfg,
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/" + slot.name,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/remote/chassis", plunger],
                    ),
                )
            )
        fields.append(
            (
                "contact_remote",
                ContactSensorCfg,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/remote/chassis",
                    filter_prim_paths_expr=[],
                ),
            )
        )
        return make_configclass("BatteryContactCfg", fields)()

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="battery_insertion",
                    predicate_sequence=[batteries_retained],
                    description="Every active cell is fully seated, correctly oriented, released, and still retained.",
                )
            ],
            failures={
                "battery_off_table": TerminationTermCfg(func=battery_off_table),
                "remote_moved": TerminationTermCfg(func=remote_moved),
                "damage_proxy": TerminationTermCfg(func=damage_proxy),
            },
        )

    def get_events_cfg(self):
        # Joint reset restores the robot home and the extended plungers. It does not weld a battery.
        # The score reset drops ever-lifted, retention timers, and hard-failure latches.
        return make_configclass(
            "BatteryEventsCfg",
            [
                (
                    "reset_joints",
                    EventTermCfg,
                    EventTermCfg(func=reset_all_articulation_joints, mode="reset"),
                ),
                (
                    "reset_score",
                    EventTermCfg,
                    EventTermCfg(func=reset_battery_insertion_state, mode="reset"),
                ),
            ],
        )()

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("battery insertion does not provide a mimic environment")

    def get_metrics(self) -> list[MetricBase]:
        return [InsertionScoreMetric(), SuccessRateMetric()]

    def get_rewards_cfg(self):
        return make_configclass(
            "BatteryRewardsCfg",
            [
                (
                    "insertion_state",
                    RewardTermCfg,
                    RewardTermCfg(func=insertion_state_tracker, weight=1.0),
                )
            ],
        )()

    def get_viewer_cfg(self) -> ViewerCfg:
        # Provisional. The smoke launch uses --visualizer none, so this is unused there.
        return ViewerCfg(eye=(1.35, -0.9, 1.25), lookat=(0.50, 0.0, 0.78))
