"""Differentiable Moreau-type time-stepping model of the stick-slip vibro-impact slider.

Each step solves the one-degree-of-freedom complementarity problem in closed form:
  friction  : v_{n+1} in v_free - (h/m) * F_k * Sgn(v_{n+1}), i.e. projection onto the box,
              with the static level F_s = r * F_k used for release from rest (r learnable, r >= 1)
  walls     : unilateral constraints at +/-0.25 m with Newton restitution e(v) = clip(e0 - a |v|),
              a >= 0 learnable, and rest-on-wall when the rebound is below a declared speed.
The class bank of the classical fitter is replaced by two continuous parameters: the threshold
ratio r (1 = coulomb, 1.6 = static) and the restitution slope a (0 = constant, 0.10 = speed).
Gradients flow through the smooth branches of every step; branch switches are piecewise
constant, so this is a sub-gradient method, not a smoothed one. Batched over runs and starts.
"""
import math
import torch

DTYPE = torch.float64
WALL, COLLAPSE, NOISE = 0.25, 1e-6, 0.003
BOUNDS = {'friction': (.16, .22), 'restitution': (.72, .82), 'spring': (2.0, 2.4), 'damping': (.14, .22), 'ratio': (1.0, 2.0), 'slope': (0.0, 0.2)}
NAMES = list(BOUNDS)


def unpack(raw):
    """raw: (..., 6) unconstrained -> dict of bounded parameters, each (...)."""
    out = {}
    for i, k in enumerate(NAMES):
        lo, hi = BOUNDS[k]; out[k] = lo + (hi - lo) * torch.sigmoid(raw[..., i])
    return out


def simulate(params, profiles, times, h=2e-3, horizon=4.0):
    """params: dict of tensors shaped (B,), profiles: (R, 4) segment forces, times: (T,) sample times.
    Returns positions (B, R, T). Mass 1."""
    B, R = params['friction'].shape[0], profiles.shape[0]
    fk, e0, k, c, r, a = (params[n][:, None].expand(B, R) for n in NAMES)
    fs = r * fk
    x = torch.zeros(B, R, dtype=DTYPE); v = torch.zeros(B, R, dtype=DTYPE)
    n_steps = int(round(horizon / h)); out = []; sample = torch.round(times / h).long().tolist(); want = {s: i for i, s in enumerate(sample)}
    for n in range(n_steps):
        t = n * h; seg = min(int(t), 3); F = profiles[:, seg][None, :].expand(B, R)
        net = F - k * x                                   # force at rest
        at_rest = v == 0
        v_free = v + h * (net - c * v)                     # semi-implicit free velocity
        level = torch.where(at_rest, fs, fk)               # static level only for release from rest
        slip_impulse = h * level
        stuck = v_free.abs() <= slip_impulse
        v_new = torch.where(stuck, torch.zeros_like(v_free), v_free - slip_impulse * torch.sign(v_free))
        x_new = x + h * v_new
        # walls: unilateral constraint with Newton restitution; sliding into the wall this step
        hit_r = (x_new > WALL) & (v_new > 0); hit_l = (x_new < -WALL) & (v_new < 0)
        hit = hit_r | hit_l
        e = torch.clamp(e0 - a * v_new.abs(), 0.2, 0.95)
        v_imp = -e * v_new
        v_imp = torch.where(v_imp.abs() < COLLAPSE, torch.zeros_like(v_imp), v_imp)
        x_new = torch.where(hit_r, torch.full_like(x_new, WALL), torch.where(hit_l, torch.full_like(x_new, -WALL), x_new))
        v_new = torch.where(hit, v_imp, v_new)
        # rest on wall: at the wall with net force pushing into it beyond the static level -> stay
        on_wall = (x_new.abs() >= WALL - 1e-12) & (v_new == 0)
        push_in = torch.sign(x_new) * net > 0
        v_new = torch.where(on_wall & push_in, torch.zeros_like(v_new), v_new)
        x, v = x_new, v_new
        if n + 1 in want: out.append(x)
    return torch.stack(out, dim=-1)


def interpolate_to(times, h):
    return times
