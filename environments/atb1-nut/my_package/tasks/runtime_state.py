"""Per-episode scorer memory. No Isaac Sim imports."""

from __future__ import annotations

from my_package.scoring.scorer import carry_episode_memory, step
from my_package.scoring.snapshot import clear_episode_state


class EpisodeRuntime:
    def __init__(self):
        self.snapshot = None
        self.last_report = None
        self.pending_clear = True
        self.token = None

    def on_reset(self) -> None:
        # Keep last_report until the next observation so the recorder can read it.
        self.pending_clear = True

    def observe(self, cfg, observed, token, dt: float):
        if token == self.token and self.last_report is not None and not self.pending_clear:
            return self.snapshot, self.last_report
        if self.pending_clear or self.snapshot is None:
            observed = clear_episode_state(observed)
            self.pending_clear = False
        else:
            observed = carry_episode_memory(self.snapshot, observed)
        self.snapshot, self.last_report = step(cfg, observed, dt)
        self.token = token
        return self.snapshot, self.last_report
