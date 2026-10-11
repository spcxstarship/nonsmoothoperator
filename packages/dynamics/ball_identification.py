"""Observation-only input and classical inverse fit for the one-rebound ball."""

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import least_squares

from .bouncing_ball import BouncingBall


@dataclass(frozen=True)
class Observations:
    initial_height: float
    acceleration: float
    times: np.ndarray
    heights: np.ndarray


def predict(floor, restitution, initial_height, acceleration, times, gravity=9.81):
    """Analytic world height and velocity, excluding the second impact."""
    ball = BouncingBall(initial_height=initial_height - floor,
                        gravity=gravity, restitution=restitution,
                        applied_acceleration=acceleration)
    times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(times >= ball.next_impact_time):
        raise ValueError("times must be finite and before the second impact")
    a, tau = ball.effective_gravity, ball.impact_time
    elapsed = times - tau
    heights = np.where(times < tau, initial_height - 0.5 * a * times**2,
                       floor + restitution * a * tau * elapsed - 0.5 * a * elapsed**2)
    velocities = np.where(times < tau, -a * times, restitution * a * tau - a * elapsed)
    return heights, velocities


def validate_observations(observations, floor_bounds, restitution_bounds, gravity):
    """Check the entire prior supports exactly one possible impact in the horizon."""
    if not observations:
        raise ValueError("at least one observed trajectory is required")
    if not (0 <= restitution_bounds[0] < restitution_bounds[1] < 1):
        raise ValueError("invalid restitution bounds")
    if not floor_bounds[0] < floor_bounds[1]:
        raise ValueError("invalid floor bounds")
    for obs in observations:
        if (obs.times.ndim != 1 or obs.times.shape != obs.heights.shape
                or len(obs.times) < 2 or not np.all(np.isfinite(obs.heights))
                or np.any(np.diff(obs.times) <= 0)):
            raise ValueError("observations must be finite, matching vectors with increasing times")
        for z, e in product(floor_bounds, restitution_bounds):
            predict(z, e, obs.initial_height, obs.acceleration, obs.times, gravity)
        latest = BouncingBall(initial_height=obs.initial_height - floor_bounds[0],
                              gravity=gravity, applied_acceleration=obs.acceleration).impact_time
        if latest >= obs.times[-1]:
            raise ValueError("each horizon must extend beyond the latest possible first impact")


def fit_classical(observations, floor_bounds, restitution_bounds, gravity=9.81):
    """Bounded Gaussian maximum likelihood, with nine fixed starting points.

    No hidden parameters, simulator seed, or true event times are inputs.
    Equal known observation variances make ordinary least squares sufficient.
    """
    validate_observations(observations, floor_bounds, restitution_bounds, gravity)

    def residual(parameters):
        return np.concatenate([predict(*parameters, obs.initial_height, obs.acceleration,
                                       obs.times, gravity)[0] - obs.heights for obs in observations])

    lower = np.array([floor_bounds[0], restitution_bounds[0]])
    upper = np.array([floor_bounds[1], restitution_bounds[1]])
    fits = [least_squares(residual, lower + np.array(fractions) * (upper - lower),
                          bounds=(lower, upper), ftol=1e-12, xtol=1e-12, gtol=1e-12,
                          max_nfev=1000)
            for fractions in product((0.2, 0.5, 0.8), repeat=2)]
    fit = min(fits, key=lambda result: np.sum(result.fun**2))
    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    return {
        "floor_m": float(fit.x[0]), "restitution": float(fit.x[1]),
        "observation_rmse_m": float(np.sqrt(np.mean(fit.fun**2))),
        "optimizer_success": bool(fit.success),
        "starts": [{"parameters": item.x.tolist(), "sse": float(np.sum(item.fun**2)),
                    "success": bool(item.success), "evaluations": item.nfev} for item in fits],
        "jacobian_singular_values": singular_values.tolist(),
    }
