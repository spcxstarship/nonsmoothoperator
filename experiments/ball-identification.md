# First ball parameter-identification pilot

## Question and scope

Can a flow/jump PINN recover floor height and restitution from noisy positions
without measured velocities or labelled impact times? Compare against a
classical analytic inverse fit to distinguish optimization problems from a lack
of information in the observations.

This is an exploratory implementation baseline, not a novel method, an active
learning study, or evidence that a PINN improves on classical identification.
All methods receive the same 36 positions. The three PINN seeds change network
initialization only; they share one noise realization, so they do not establish
robustness across measurement noise or hidden physical parameters.

## Reproduce

Use Python 3.12 and install the pinned dependencies into a virtual environment:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-pinn.txt
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python experiments/identify_ball.py --out data/runs/ball-identification-v1
.venv/bin/python experiments/summarize_ball.py data/runs/ball-identification-v1
```

Choose a fresh output directory for every run. The runner rejects an existing
directory and saves its exact configuration and observation file before fitting.
Generated results and checkpoints stay under ignored `data/`.

`report.json` contains every fitted parameter, diagnostic, timing, and pass/fail
flag. `identification.png` shows the observations and all three neural fits.
Per-seed folders contain checkpoints, raw training histories, and dense curves.
The summary command preserves the original report and writes a supplementary
`integrated-evaluation.json` alongside the plot.

The fixed [configuration](ball-identification.json) specifies observations,
parameter bounds, network size, loss weights, optimizer budgets, evaluation
inputs, and pilot success criteria. It was written before the first training
run. Thresholds are engineering checks for this pilot, not statistical tests.
Any later change to the setup should get a distinct run directory and preserve
the original result.

## Equations and available information

World height is `h`, vertical velocity is `v`, floor height is `z`, and
restitution is `e`. The flow is

```text
dh/dt = v
dv/dt = -g + u
```

At `h = z`, with downward velocity, the reset is

```text
h_post = z
v_post = -e * v_pre
```

The generator uses the existing `BouncingBall` reference, offset by `z`.
Observations are `y_k = h(t_k) + noise_k`, with independent Gaussian noise of
standard deviation 0.01 m. Initial height (1 m), zero initial velocity, gravity,
actuation, observation times, and the model family are known. Only `z` and `e`
are unknown physical parameters. Each of the three trajectories uses a constant
acceleration of -4, 0, or +4 m/s² and twelve fixed observations through 0.70 s.

The inference prior is `z in [0, 0.25] m`, `e in [0.6, 0.95]`. These bounds imply
exactly one impact within each training horizon and exclude a second impact.
This known event count is a substantial simplifying assumption. The pilot does
not discover whether contact occurs or how many events occur.

The fitting functions receive observation objects, inference settings, and known
gravity. The generator seed, true parameters, true velocities, and true event
times are absent from their inputs. `observations.json` also omits these values.
The full configuration and evaluation files contain synthetic ground truth for
reproducibility and must not be supplied to a future experiment-selection agent.

## Classical estimator

`packages/dynamics/ball_identification.py` minimizes position squared error with
SciPy's bounded least squares, using nine fixed starting points. With equal
Gaussian observation variances this is maximum likelihood. Every start is
reported; the smallest observation loss selects the result. The singular values
of the observation Jacobian are saved as a local sensitivity diagnostic, not a
global identifiability guarantee.

## Flow/jump PINN

`packages/pinn/ball.py` uses separate pre/post networks for each training
trajectory. Each network has two tanh hidden layers of width 16 and outputs
height and velocity terms. All computations use float64 on one CPU thread.

For each trajectory, learn a latent event time `tau` and normalize segment time:

```text
s = t / tau                 before impact
r = (t - tau) / (T - tau)    after impact
```

The two physical parameters are shared by all trajectories. Sigmoid transforms
keep them within the prior. Event times have independent trainable parameters,
bounded by `sqrt(2*(h0-z_max)/(g-u))` and `sqrt(2*(h0-z_min)/(g-u))`. Only the
prior bounds enter these expressions; true impact times are never supplied.
The exact event-time formula for the current fitted floor is not imposed.

Initial height and velocity, post-impact height, and the restitution reset are
hard constraints in the neural parameterization. A loss enforces the pre-impact
height meeting the floor. Flow residuals are evaluated by automatic
differentiation on 48 fixed coordinates per segment; none cross a jump.
The loss is

```text
L = L_position + 100 * L_flow + 100 * (L_guard + L_admissibility)
```

Position quantities use a 1 m scale. The two flow residuals are normalized by
`sqrt(g * 1 m)` and `g` respectively before squaring. The admissibility penalty
discourages floor penetration and upward pre-impact velocity. Averages over
trajectories/segments are explicit in the implementation. These are finite
penalties, not exact physical guarantees.

Training uses 1,500 Adam steps followed by at most 1,000 L-BFGS iterations with
strong-Wolfe line search. All three initializations and any numerical failures
are retained. No classical fitted parameters initialize the PINN. Loss history,
closure calls, wall time, final diagnostics, and weights are saved per seed.

## Evaluation

Three distinct evaluations are reported after fitting:

1. **Physical parameter recovery:** absolute errors in `z` and `e`.
2. **Neural reconstruction:** dense unobserved positions on the same three
   training trajectories, and their learned event-time errors. Physical
   residuals use 197 independent midpoint coordinates per segment.
3. **Parameter transfer:** plug each method's fitted parameters into the same
   analytic simulator and predict four fresh trajectories (initial heights 0.85
   and 1.15 m, accelerations -2 and +2 m/s²). Report height/velocity RMSE and
   impact-time MAE through 0.62 s. All priors remain before a second impact.

The original report's RMSE uses the fixed time grid. Velocity jumps make this
metric sensitive to whether a grid point lies in a short event-time mismatch.
The summary script additionally integrates squared height and velocity error,
splitting the integration at both true and estimated impact times. Within each
interval these errors are polynomials; three-point Gauss-Legendre quadrature
integrates their squares exactly up to floating-point precision. This check
does not retrain models, select seeds, or change the predeclared pilot criteria.

Parameter transfer is not direct neural-network generalization: the fitted
trajectory networks have no representation for new initial conditions or inputs.
A later conditional PINN would be needed to establish that capability. Dense
neural reconstruction is interpolation on fitted trajectories, not an independent
trajectory test. The reserved evaluation trajectories and synthetic truth do
not enter optimization or model selection.

The classical fit and PINN have equal data and physical assumptions, but different
function classes and compute budgets. This is a feasibility comparison. Save
their compute costs rather than describing it as an equal-compute contest.
