"""Nut-threading task terms. The reward term only advances the scorer."""

from __future__ import annotations

from functools import partial

from isaaclab_arena.tasks.task_base import TaskBase


class NutThreadingTask(TaskBase):
    def __init__(self, cfg):
        super().__init__(episode_length_s=cfg.episode_length_s, task_description=cfg.task_description)
        self.cfg = cfg

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        from isaaclab.managers import EventTermCfg
        from isaaclab_arena.utils.configclass import make_configclass

        from my_package.tasks.sim_bridge import reset_episode_memory

        term = EventTermCfg(func=reset_episode_memory, mode="reset")
        return make_configclass("NutEventsCfg", [("reset_nut_memory", EventTermCfg, term)])()

    def get_termination_cfg(self):
        from isaaclab.managers import TerminationTermCfg
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

        from my_package.tasks.sim_bridge import failure_fixture_moved, failure_off_table, success_fully_seated

        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="nut_seated",
                    predicate_sequence=[partial(success_fully_seated, cfg=self.cfg)],
                )
            ],
            failures={
                "off_table": TerminationTermCfg(func=failure_off_table, params={"cfg": self.cfg}),
                "fixture_moved": TerminationTermCfg(func=failure_fixture_moved, params={"cfg": self.cfg}),
            },
        )

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("nut threading has no mimic environment")

    def get_metrics(self):
        from isaaclab_arena.metrics.success_rate import SuccessRateMetric

        from my_package.tasks.sim_bridge import make_nut_score_metric

        return [make_nut_score_metric(), SuccessRateMetric()]

    def get_rewards_cfg(self):
        from isaaclab.managers import RewardTermCfg
        from isaaclab_arena.utils.configclass import make_configclass

        from my_package.tasks.sim_bridge import advance_score_reward

        term = RewardTermCfg(func=advance_score_reward, weight=0.0, params={"cfg": self.cfg})
        return make_configclass("NutRewardsCfg", [("advance_nut_score", RewardTermCfg, term)])()

    def get_viewer_cfg(self):
        from isaaclab.envs.common import ViewerCfg

        front = next(camera for camera in self.cfg.cameras if camera.name == "front")
        return ViewerCfg(eye=front.eye_xyz, lookat=front.target_xyz)
