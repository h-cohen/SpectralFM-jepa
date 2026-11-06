"""Early stopping on a validation metric (lower is better). Labels never enter: the metric is a
pretraining loss on held-out spectra. The state is saved in every checkpoint so a resume continues it."""
from __future__ import annotations

import math


class EarlyStopping:
    def __init__(self, metric: str, patience: int, min_delta: float, start_step: int):
        if patience < 1:
            raise ValueError(f"training.early_stopping.patience={patience} must be >= 1")
        self.metric, self.patience, self.min_delta, self.start_step = metric, patience, min_delta, start_step
        self.best_value: float | None = None
        self.best_step: int | None = None
        self.bad_checks = 0

    def update(self, step: int, metrics: dict) -> bool:
        """Record one validation check; True if it is a new best. Checks before start_step are ignored."""
        if self.metric not in metrics:
            raise ValueError(f"training.early_stopping.metric {self.metric!r} not in {sorted(metrics)}")
        if step < self.start_step:
            return False
        value = float(metrics[self.metric])
        if math.isfinite(value) and (self.best_value is None or value < self.best_value - self.min_delta):
            self.best_value, self.best_step, self.bad_checks = value, step, 0
            return True
        self.bad_checks += 1   # NaN/inf never improves
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_checks >= self.patience

    def state_dict(self) -> dict:
        return {"best_value": self.best_value, "best_step": self.best_step, "bad_checks": self.bad_checks}

    def load_state_dict(self, state: dict) -> None:
        self.best_value, self.best_step, self.bad_checks = state["best_value"], state["best_step"], state["bad_checks"]
