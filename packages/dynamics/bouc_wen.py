"""Dimensionless, normalized Bouc-Wen oscillator for an exploratory pilot.

This is an adaptation, NOT a reproduction of the 2016 benchmark parameters.
State (x, v, w): x'=v; v'=u-c*v-k*x-h*w;
w'=rho*(v-sigma*abs(v)*w-(1-sigma)*v*abs(w)).
Mass=1, exponent n=1, sigma=0.75 and initial state=0 are known.
Only (c, k, h, rho) are estimated; the hidden-state scale is fixed.

Benchmark motivation: https://doi.org/10.4121/12967592
The normalized equation is stated explicitly here rather than relying on
incompatible alpha/beta/gamma conventions across publications.
"""

from math import pi, sin, sqrt

import numpy as np
from scipy.integrate import solve_ivp


PARAMETERS = ("damping", "stiffness", "hysteretic_gain", "memory_rate")
LOWER = np.array([0.10, 0.60, 0.30, 0.60])
UPPER = np.array([0.40, 1.40, 1.10, 2.00])
DURATION = 16.0
SIGMA = 0.75
CANDIDATES = ("slow_cycles", "fast_cycles", "mixed_cycles")
RESERVED = ("chirp", "nested_cycles")
PROFILES = ("baseline", *CANDIDATES, *RESERVED)
OBS_TIMES = np.linspace(0.25, DURATION, 64)


def forcing(profile, t):
    """Candidates have equal duration and analytic RMS force sqrt(0.32).

Their peak forces differ, but all are below 1.4. Frequency and waveform
effects are inseparable in this first screening comparison.
"""
    slow = sin(2 * pi * t / 8)
    fast = sin(2 * pi * t / 4)
    if profile == "baseline":
        return 0.08 * slow
    if profile == "slow_cycles":
        return 0.8 * slow
    if profile == "fast_cycles":
        return 0.8 * fast
    if profile == "mixed_cycles":
        return 0.8 * (slow + fast + sin(2 * pi * t / 2)) / sqrt(3)
    if profile == "chirp":
        return 0.8 * sin(2 * pi * (0.06 * t + 0.003 * t * t))
    if profile == "nested_cycles":
        return 0.6 * sin(2 * pi * t / 10) + 0.3 * sin(2 * pi * t / 3.3)
    raise ValueError(f"Unknown profile: {profile}")


def simulate(parameters, profile, times=OBS_TIMES, *, reference=False,
             initial=(0.0, 0.0, 0.0)):
    """Return all three states; inference must use only column zero.

The reference uses a separately written RHS, RK45, tighter tolerances and
smaller steps. It checks numerical agreement, not independent model validity.
"""
    p = np.asarray(parameters, dtype=float)
    t = np.asarray(times, dtype=float)
    if p.shape != (4,) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Four nonnegative finite parameters required")
    if profile not in PROFILES:
        raise ValueError("Unknown profile")
    if (t.ndim != 1 or not len(t) or not np.all(np.isfinite(t))
            or t[0] < 0 or t[-1] > DURATION or np.any(np.diff(t) <= 0)):
        raise ValueError("Times must be strictly increasing in [0,16]")
    state = np.asarray(initial, dtype=float)
    if state.shape != (3,) or not np.all(np.isfinite(state)):
        raise ValueError("Three finite initial coordinates required")
    c, k, h, rho = p

    def rhs(time, y):
        x, v, w = y
        return (v, forcing(profile, time) - c*v - k*x - h*w,
                rho*(v - SIGMA*abs(v)*w - (1-SIGMA)*v*abs(w)))

    def reference_rhs(time, y):
        displacement, speed, memory = y
        # Same law rearranged by the direction of motion.
        memory_slope = (1 - SIGMA*np.sign(speed)*memory
                        - (1-SIGMA)*abs(memory))
        return [speed,
                forcing(profile, time) - np.dot(p[:3], [speed, displacement, memory]),
                p[3]*speed*memory_slope]

    if t[-1] == 0:
        return state[None, :]
    sol = solve_ivp(reference_rhs if reference else rhs, (0, float(t[-1])), state,
                    t_eval=t, method="RK45" if reference else "DOP853",
                    rtol=1e-10 if reference else 2e-7,
                    atol=1e-12 if reference else 2e-9,
                    max_step=0.025 if reference else 0.25)
    if not sol.success or sol.y.shape != (3, len(t)) or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(f"Bouc-Wen integration failed: {sol.message}")
    return sol.y.T
