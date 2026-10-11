"""Analytic bouncing-ball reference through the first rebound.

The impact has two states at the same time: ``side="pre"`` and
``side="post"``. Keeping both states avoids smoothing away the velocity jump.
"""

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class BallState:
    height: float
    velocity: float


@dataclass(frozen=True)
class BouncingBall:
    initial_height: float = 1.0
    gravity: float = 9.81
    restitution: float = 0.8
    applied_acceleration: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.initial_height) or self.initial_height <= 0:
            raise ValueError("initial_height must be positive")
        if not isfinite(self.gravity) or self.gravity <= 0:
            raise ValueError("gravity must be positive")
        if not isfinite(self.applied_acceleration) or self.applied_acceleration >= self.gravity:
            raise ValueError("applied_acceleration must be below gravity for an impact")
        if not isfinite(self.restitution) or not 0 <= self.restitution < 1:
            raise ValueError("restitution must be in [0, 1)")

    @property
    def effective_gravity(self) -> float:
        return self.gravity - self.applied_acceleration

    @property
    def impact_time(self) -> float:
        return sqrt(2 * self.initial_height / self.effective_gravity)

    @property
    def impact_speed(self) -> float:
        return sqrt(2 * self.effective_gravity * self.initial_height)

    @property
    def next_impact_time(self) -> float:
        return self.impact_time + 2 * self.restitution * self.impact_speed / self.effective_gravity

    def state_at(self, time: float, *, side: str = "post") -> BallState:
        """Return the exact state from release through the first rebound.

        At the first impact, ``side`` chooses the state immediately before or
        after the reset. Times after the next impact are outside this benchmark.
        """
        if time < 0 or time > self.next_impact_time or (
            time == self.next_impact_time and time != self.impact_time
        ):
            raise ValueError("time is outside the first-rebound interval")
        if side not in {"pre", "post"}:
            raise ValueError("side must be 'pre' or 'post'")

        if time < self.impact_time or (time == self.impact_time and side == "pre"):
            return BallState(
                self.initial_height - 0.5 * self.effective_gravity * time**2,
                -self.effective_gravity * time,
            )

        since_impact = time - self.impact_time
        upward_speed = self.restitution * self.impact_speed
        return BallState(
            upward_speed * since_impact - 0.5 * self.effective_gravity * since_impact**2,
            upward_speed - self.effective_gravity * since_impact,
        )
