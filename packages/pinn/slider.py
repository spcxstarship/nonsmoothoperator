"""Regularised differential-inclusion PINN for the stick-slip vibro-impact slider.

One tanh network per observed run represents x(t) with x(0) = v(0) = 0 enforced by
construction. Friction is a smooth Stribeck-type function with kinetic level F_k and
static level F_s = F_k or 1.6 F_k according to the declared class; walls are stiff
Kelvin-Voigt penalties whose damping ratio encodes the restitution e_0. The loss is the
standardised observation error plus the mean squared ODE residual at collocation points.
Physical parameters (friction, restitution, spring, damping) are learned jointly through
bounded sigmoid reparameterisation. The smoothing constants are declared model choices.
Speed-dependent restitution is not represented in the residual; it enters only through
the class used when the learned parameters are evaluated in the event-driven simulator.
"""
import math
import time
import numpy as np
import torch
from torch import nn

DTYPE = torch.float64
BOUNDS = {'friction': (.16, .22), 'restitution': (.72, .82), 'spring': (2.0, 2.4), 'damping': (.14, .22)}
WALL, HORIZON, NOISE = .25, 4.0, .003
SMOOTHING = {'friction_velocity_scale': .01, 'stribeck_velocity_scale': .05, 'wall_stiffness': 1e4}


def network(width):
    return nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1)).double()


class SliderPINN(nn.Module):
    def __init__(self, history, profiles, label, width=32, seed=0, residual_weight=1.0, wall_free_residual=False, wall_margin=0.24):
        super().__init__()
        torch.manual_seed(seed)
        self.residual_weight, self.wall_free_residual, self.wall_margin = residual_weight, wall_free_residual, wall_margin
        self.history, self.profiles, self.label = history, profiles, label
        self.static_ratio = 1.6 if label.startswith('static') else 1.0
        self.nets = nn.ModuleList([network(width) for _ in history])
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.zeros((), dtype=DTYPE)) for k in BOUNDS})

    def parameter(self, key):
        lo, hi = BOUNDS[key]
        return lo + (hi - lo) * torch.sigmoid(self.raw[key])

    def parameters_dict(self):
        return {k: float(self.parameter(k)) for k in BOUNDS}

    def position(self, i, t):
        s = t / HORIZON
        return WALL * s ** 2 * self.nets[i](s[:, None])[:, 0]

    def force(self, i, t):
        profile = torch.as_tensor(self.profiles[self.history[i]['candidate_id']], dtype=DTYPE)
        index = torch.clamp(torch.floor(t).long(), 0, 3)
        return profile[index]

    def residual(self, i, t):
        t = t.clone().requires_grad_(True)
        x = self.position(i, t)
        v = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
        a = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        fk, e0, k, c = (self.parameter(n) for n in ['friction', 'restitution', 'spring', 'damping'])
        fs = self.static_ratio * fk
        eps, vs, kw = SMOOTHING['friction_velocity_scale'], SMOOTHING['stribeck_velocity_scale'], SMOOTHING['wall_stiffness']
        friction = (fk + (fs - fk) * torch.exp(-(v / vs) ** 2)) * torch.tanh(v / eps)
        zeta = -torch.log(e0) / torch.sqrt(math.pi ** 2 + torch.log(e0) ** 2)
        cw = 2 * zeta * math.sqrt(kw)
        penetration = torch.relu(x - WALL) - torch.relu(-WALL - x)
        in_contact = (penetration != 0).to(DTYPE)
        wall = kw * penetration + cw * v * in_contact
        return a + c * v + k * x + friction + wall - self.force(i, t)

    def loss(self, collocation):
        data = 0.
        times = torch.linspace(.15, 4., 16, dtype=DTYPE)
        for i, run in enumerate(self.history):
            y = torch.as_tensor(run['positions'], dtype=DTYPE)
            data = data + torch.mean((self.position(i, times) - y) ** 2) / NOISE ** 2
        residual = 0.
        for i in range(len(self.history)):
            r = self.residual(i, collocation)
            if self.wall_free_residual:
                # Variant B: enforce the flow equation only where the network predicts no wall contact;
                # impacts are then carried by the data term alone (restitution is not constrained by the residual).
                mask = (self.position(i, collocation).detach().abs() < self.wall_margin).to(DTYPE)
                residual = residual + torch.sum(mask * r ** 2) / torch.clamp(mask.sum(), min=1.)
            else:
                residual = residual + torch.mean(r ** 2)
        return data + self.residual_weight * residual, float(data), float(residual)


def train(history, profiles, label, seed=0, adam_steps=2000, lbfgs_steps=300, collocation=512, width=32, residual_weight=1.0, wall_free_residual=False):
    torch.set_num_threads(1)
    model = SliderPINN(history, profiles, label, width, seed, residual_weight, wall_free_residual)
    rng = np.random.default_rng(seed)
    points = torch.as_tensor(np.sort(rng.uniform(0., HORIZON, collocation)), dtype=DTYPE)
    points = points[(points % 1.0 > 2e-3) & (points % 1.0 < 1 - 2e-3)]  # keep away from force switches
    started = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trace = []
    for step in range(adam_steps):
        optimizer.zero_grad()
        total, data, residual = model.loss(points)
        total.backward()
        optimizer.step()
        if step % 200 == 0:
            trace.append({'step': step, 'total': float(total), 'data': data, 'residual': residual, **model.parameters_dict()})
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        lbfgs.zero_grad()
        total, _, _ = model.loss(points)
        total.backward()
        return total
    lbfgs.step(closure)
    total, data, residual = model.loss(points)
    trace.append({'step': 'lbfgs', 'total': float(total), 'data': data, 'residual': residual, **model.parameters_dict()})
    dense = torch.linspace(0., HORIZON, 401, dtype=DTYPE)
    reconstruction = [model.position(i, dense).detach().numpy().tolist() for i in range(len(history))]
    return {'parameters': model.parameters_dict(), 'class': label, 'final_loss': {'total': float(total), 'data': data, 'residual': residual},
            'trace': trace, 'dense_times': dense.numpy().tolist(), 'reconstruction': reconstruction,
            'elapsed_s': time.monotonic() - started, 'settings': {'adam_steps': adam_steps, 'lbfgs_steps': lbfgs_steps, 'collocation': int(len(points)), 'width': width, 'seed': seed, 'residual_weight': residual_weight, 'wall_free_residual': wall_free_residual, **SMOOTHING}}
