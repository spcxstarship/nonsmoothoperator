"""Piecewise trajectory PINN with latent event times and shared contact parameters.

Each observed trajectory has two tanh networks representing (height, velocity).
Initial conditions and the velocity reset are enforced by construction. The
pre-impact height equals the learned floor through a guard loss. Event times
are independently optimized within bounds implied by the parameter prior; the
analytic event-time relation is NOT substituted into the neural model.

Networks fit the observed trajectories only. Generalization to new inputs is
evaluated separately by simulating the learned physical parameters.
"""

import math
import time
import numpy as np
import torch
from torch import nn

from packages.dynamics.ball_identification import validate_observations


DTYPE = torch.float64


def tensor(value):
    return torch.as_tensor(value, dtype=DTYPE)


def network(width):
    return nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width),
                         nn.Tanh(), nn.Linear(width, 2)).double()


class BallPINN(nn.Module):
    def __init__(self, observations, inference, config, gravity=9.81, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.observations = observations
        self.floor_bounds = inference["floor_bounds_m"]
        self.restitution_bounds = inference["restitution_bounds"]
        validate_observations(observations, self.floor_bounds, self.restitution_bounds, gravity)
        self.gravity, self.velocity_scale = gravity, math.sqrt(gravity)
        self.config = config
        self.pre = nn.ModuleList([network(config["width"]) for _ in observations])
        self.post = nn.ModuleList([network(config["width"]) for _ in observations])

        def logit_initial(value, bounds):
            fraction = (value - bounds[0]) / (bounds[1] - bounds[0])
            if not 0 < fraction < 1:
                raise ValueError("initial parameter must lie strictly inside bounds")
            return math.log(fraction / (1 - fraction))

        self.raw_floor = nn.Parameter(tensor(logit_initial(inference["initial_floor_m"], self.floor_bounds)))
        self.raw_restitution = nn.Parameter(tensor(logit_initial(
            inference["initial_restitution"], self.restitution_bounds)))
        self.raw_times = nn.Parameter(tensor([math.log(4.0)] * len(observations)))
        tau_bounds = [[math.sqrt(2 * (obs.initial_height - z) / (gravity - obs.acceleration))
                       for z in reversed(self.floor_bounds)] for obs in observations]
        self.register_buffer("tau_bounds", tensor(tau_bounds))

    @property
    def floor(self):
        low, high = self.floor_bounds
        return low + (high - low) * self.raw_floor.sigmoid()

    @property
    def restitution(self):
        low, high = self.restitution_bounds
        return low + (high - low) * self.raw_restitution.sigmoid()

    @property
    def event_times(self):
        return self.tau_bounds[:, 0] + self.raw_times.sigmoid() * (
            self.tau_bounds[:, 1] - self.tau_bounds[:, 0])

    def segment(self, index, coordinate, post=False):
        """Height in m and velocity in m/s; coordinate runs from zero to one."""
        if post:
            raw = self.post[index](coordinate)
            _, before_velocity = self.segment(index, torch.ones_like(coordinate))
            return (self.floor + coordinate * raw[:, :1],
                    -self.restitution * before_velocity + self.velocity_scale * coordinate * raw[:, 1:])
        raw = self.pre[index](coordinate)
        return (self.observations[index].initial_height + coordinate**2 * raw[:, :1],
                self.velocity_scale * coordinate * raw[:, 1:])

    def forward(self, index, times):
        times = tensor(times).reshape(-1, 1)
        tau = self.event_times[index]
        horizon = self.observations[index].times[-1]
        before = self.segment(index, times / tau)
        after = self.segment(index, (times - tau) / (horizon - tau), post=True)
        return tuple(torch.where(times < tau, a, b) for a, b in zip(before, after))

    def residuals(self, index, coordinate, post=False):
        coordinate = coordinate.detach().clone().requires_grad_(True)
        height, velocity = self.segment(index, coordinate, post)
        duration = self.event_times[index]
        if post:
            duration = self.observations[index].times[-1] - duration
        dh = torch.autograd.grad(height.sum(), coordinate, create_graph=True)[0] / duration
        dv = torch.autograd.grad(velocity.sum(), coordinate, create_graph=True)[0] / duration
        a = self.gravity - self.observations[index].acceleration
        return dh - velocity, dv + a, height

    def loss_terms(self):
        grid = torch.linspace(0, 1, self.config["collocation_points_per_segment"], dtype=DTYPE)[:, None]
        data, flow, guard, admissibility = [], [], [], []
        for index, obs in enumerate(self.observations):
            predicted, _ = self(index, obs.times)
            data.append((predicted[:, 0] - tensor(obs.heights)).square().mean())
            hpre, vpre = self.segment(index, tensor([[1.0]]))
            guard.append((hpre - self.floor).square().mean())
            admissibility.append((vpre / self.velocity_scale).relu().square().mean())
            for post in (False, True):
                dh, dv, height = self.residuals(index, grid, post)
                flow.append((dh / self.velocity_scale).square().mean() + (dv / self.gravity).square().mean())
                admissibility.append((self.floor - height).relu().square().mean())
        return {"data": torch.stack(data).mean(), "flow": torch.stack(flow).mean(),
                "guard": torch.stack(guard).mean(), "admissibility": torch.stack(admissibility).mean()}

    def loss(self):
        terms = self.loss_terms()
        return (terms["data"] + self.config["physics_weight"] * terms["flow"]
                + self.config["guard_weight"] * (terms["guard"] + terms["admissibility"]))

    def diagnostics(self, points=197):
        # Midpoints are independent of the training collocation grid.
        grid = ((torch.arange(points, dtype=DTYPE) + 0.5) / points)[:, None]
        dh_values, dv_values, guards = [], [], []
        for index in range(len(self.observations)):
            hpre, _ = self.segment(index, tensor([[1.0]]))
            guards.append(abs(float((hpre - self.floor).detach())))
            for post in (False, True):
                dh, dv, _ = self.residuals(index, grid, post)
                dh_values.extend(dh.detach().numpy().ravel())
                dv_values.extend(dv.detach().numpy().ravel())
        return {
            "flow_dh_minus_v_rms_m_s": float(np.sqrt(np.mean(np.square(dh_values)))),
            "flow_dv_plus_effective_gravity_rms_m_s2": float(np.sqrt(np.mean(np.square(dv_values)))),
            "guard_absolute_error_m": max(guards),
            "loss_terms": {key: float(value.detach()) for key, value in self.loss_terms().items()},
        }


def train(model, progress=None):
    started = time.perf_counter()
    history = []
    adam = torch.optim.Adam(model.parameters(), lr=model.config["adam_learning_rate"])
    for step in range(model.config["adam_steps"]):
        adam.zero_grad()
        loss = model.loss()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite PINN loss")
        loss.backward()
        adam.step()
        if step % 250 == 0 or step + 1 == model.config["adam_steps"]:
            row = {"optimizer": "adam", "step": step + 1, "loss": float(loss.detach())}
            history.append(row)
            if progress:
                progress(row)

    calls = 0
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=model.config["lbfgs_steps"],
                                 max_eval=2 * model.config["lbfgs_steps"],
                                 tolerance_grad=1e-10, tolerance_change=1e-13,
                                 history_size=50, line_search_fn="strong_wolfe")

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        loss = model.loss()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite PINN loss")
        loss.backward()
        calls += 1
        if calls % 250 == 0:
            row = {"optimizer": "lbfgs", "evaluation": calls, "loss": float(loss.detach())}
            history.append(row)
            if progress:
                progress(row)
        return loss

    optimizer.step(closure)
    return {"elapsed_s": time.perf_counter() - started, "history": history,
            "lbfgs_closure_calls": calls, "final_loss": float(model.loss().detach()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters())}
