# The stick-slip slider as a hybrid dynamical system

This note writes the stick-slip vibro-impact benchmark (`experiments/stick_slip_offgrid.py`,
`simulate`) in the hybrid-inclusion form H = (C, F, D, G) used in the ELE4023 notes and in
Goebel, Sanfelice and Teel (2012), and records how the simulator realises each piece.
It is an original synthetic system, not a replication of a published benchmark.

## Physical model

A unit mass at position x in [-0.25, 0.25] m, velocity v, attached to a linear spring of
stiffness k at the origin, with viscous damping c, dry friction with kinetic level F_k and
static level F_s, driven by a piecewise-constant force u(t) (four 1-second segments). Rigid
walls at x = +/-0.25 m apply an instantaneous velocity reversal with restitution e.

Mechanism classes (unknown to the estimator; four combinations):

| Axis | Option A | Option B |
| --- | --- | --- |
| Stick threshold | coulomb: F_s = F_k | static: F_s = 1.6 F_k |
| Restitution | constant: e = e_0 | speed: e = clip(e_0 - 0.10 |v^-|, 0.2, 0.95) |

Unknown continuous parameters theta = (F_k, e_0, k, c) with bounds
F_k in [0.16, 0.22] N, e_0 in [0.72, 0.82], k in [2.0, 2.4] N/m, c in [0.14, 0.22] N s/m.

## Hybrid state and modes

State z = (x, v, q, tau) with discrete mode q in {slip+, slip-, stick, wall_rest} and a clock
tau in [0, 4] that selects the force segment. The mode is not observed. Only x(t_i) at 16
fixed times t_i (0.15 s to 4.0 s) is observed, with Gaussian noise of standard deviation 3 mm.

Let n(z, t) = u(t) - k x be the net applied-plus-spring force at rest.

**Flow set C and flow map F.**

- slip+/-: v != 0 with sign(v) = +/-1, |x| <= 0.25:
  x' = v, v' = u(t) - k x - c v - F_k sign(v), tau' = 1.
- stick: v = 0, |x| < 0.25, |n| <= F_s: x' = 0, v' = 0, tau' = 1.
- wall_rest: v = 0, |x| = 0.25, and the net force pushes into the wall or is below F_s:
  x' = 0, v' = 0, tau' = 1.

Within a slip segment the flow is an exactly solvable damped linear oscillator about the shifted
equilibrium x_eq = (u - F_k sign(v)) / k; the simulator uses this closed form.

**Jump set D and jump map G.**

| Event (guard) | Condition in D | Reset G |
| --- | --- | --- |
| wall impact | |x| = 0.25, v directed into the wall | v^+ = -e(v^-) v^-, x^+ = x |
| stop | v reaches 0 away from a wall | q^+ = stick, v^+ = 0 |
| release | q = stick and |n| > F_s (only possible at a force-segment boundary, since u is piecewise constant and x fixed) | q^+ = slip with sign(v) = sign(n) |
| wall release | q = wall_rest and the net force pulls off the wall by more than F_s | q^+ = slip inward |
| contact collapse | post-impact |v^+| < 1e-6 m/s | v^+ = 0, q^+ = wall_rest |

Hybrid time (t, j): t is physical time, j counts jumps. Solutions are hybrid arcs on a compact
hybrid time domain over t in [0, 4]. The last row is the engineering treatment of Zeno
accumulation: as in the bouncing ball of the lecture notes, repeated impacts under a constant
inward force accumulate in finite time; the simulator declares contact once the rebound speed
falls below 1e-6 m/s and continues with the wall_rest flow. The event cap of 300 per run and the
"nonadvancing event" guard protect against numerically Zeno sequences that the collapse rule
does not catch.

## What is identifiable from which experiment

- The stick threshold F_s is only exercised when a stuck mass receives a force with
  F_k < |n| < 1.6 F_k. The staircase profile (0.18, 0.26, 0.34, 0.42 N) is designed to do this.
- Restitution and its speed dependence are only exercised by impacts at distinct speeds.
- Spring and damping are best exercised by free, unforced decay between walls.

This is why a three-phase schedule (staircase, high-speed reversal, push-and-release) matches
the best observed experiment selector, and why a local Fisher criterion that never leaves the
impact regime cannot identify the stick-threshold class.

## Representations for a PINN

1. **Flow/jump (segment) PINN.** One network per flow segment with learned event times and the
   reset enforced by construction, as in `packages/pinn/ball.py`. Requires the number and order
   of events, so it does not scale to this system where the event sequence is unknown.
2. **Regularised differential-inclusion PINN.** Replace the set-valued friction law by a smooth
   Stribeck-type function and the rigid wall by a stiff Kelvin-Voigt penalty whose damping ratio
   encodes e. A single network x(t) per run, ODE residual by autograd, physical parameters
   learned jointly. Implemented in `packages/pinn/slider.py`. Cheap and class-agnostic in
   structure, but smoothing biases the stick threshold and a smooth network represents impacts
   poorly.
3. **Complementarity (time-stepping) PINN.** Treat the friction force and the contact impulse as
   complementarity variables with the gap and the slip velocity; the mode sequence is then
   implicit in the constraint. This is the representation the project specification singles
   out and the one that should follow this pilot.
