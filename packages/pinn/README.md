# PINN

Physics-informed neural network models, residuals, losses, and training components.

`ball.py` implements the first flow/jump inverse PINN. It jointly learns floor
height, restitution, and one latent impact time per observed trajectory, using
two small neural state representations per trajectory. See the
[experiment protocol](../../experiments/ball-identification.md) for equations,
limitations, reproducible commands, and evaluation definitions.

## Research target

Develop a reusable physics-informed neural framework for hybrid dynamical
systems, and establish its accuracy, reliability, and computational cost across
distinct system families. Broad applicability is the objective; universal
superiority over every possible hybrid system is not an established or
testable benchmark claim.

The initial supported class should be finite-dimensional, deterministic,
piecewise-smooth systems with a declared mode graph, flow equations, event
conditions, and reset rules. Start with well-posed trajectories and finitely
many separated events over the evaluation horizon. Simultaneous events,
grazing, Zeno accumulation, stochastic dynamics, and unknown mechanism discovery
require explicit extensions and separate validation.

## Shared physics interface

Each system adapter should supply:

- State and parameter dimensions, units, scales, and parameter bounds.
- Flow residuals within each mode, including known inputs.
- Guard conditions and permitted transitions between modes.
- Reset maps or reset residuals, including which state components may jump.
- Initial conditions, admissibility constraints, and the observation model.

Physics residuals need a computation path suitable for the chosen training
method; access to a black-box simulator alone is not a complete PINN interface.
The reference simulator should remain separate from the neural model and serve
as a data generator and evaluator. Fitting and selection must not receive
hidden true parameters or evaluator event labels.

## Model and training development

The first candidate architecture is a piecewise neural state representation
with explicit event times and reset constraints. Evaluate flow residuals within
smooth segments and evaluate jumps through their own constraints. A repeated
visit to a mode must be supported; two segments per trajectory is not a general
hybrid representation.

Separate two tasks and their evidence:

1. **Inverse identification:** infer parameters, event times, and latent states
   from sparse noisy observations. First validate a supplied event sequence;
   then test event-count and mode-sequence inference as distinct extensions.
2. **Conditional prediction:** train a neural model that accepts initial
   conditions, inputs, and appropriate physical parameters, and evaluate its
   direct predictions on unseen conditions. Analytic rollouts using fitted
   parameters do not establish neural generalization.

Compare representation choices and training changes through controlled
ablations. Candidate changes include hard versus penalized constraints,
dimensionless residual scaling, loss balancing, temporal training schedules,
and collocation near events. These are hypotheses to evaluate, not a claim that
one combination is already best. Keep experiment selection fixed during the
initial architecture comparisons so model and data-selection effects can be
distinguished.

## Evidence required

Use an unchanged core trainer across a progression of benchmarks: repeated
bouncing-ball impacts, switching flow without a state jump, and stick-slip with
impacts. Freeze development and confirmation splits by physical parameters,
initial conditions, inputs, and noise streams. Report network initialization
variability separately from data variability.

Compare a smooth PINN, explicit flow/jump variants, and classical identification
or numerical reference methods appropriate to each task. Match observation
budgets, bound tuning and training budgets, and report actual computation.
Measure direct neural state error, event timing and sequence errors, reset and
admissibility violations, parameter recovery, and failure rates. Retain failed
runs and report per-system outcomes alongside any aggregate score.

The next implementation milestone is one generic finite-event trainer running
both the existing ball example and a switching-flow example, with known event
sequence and hidden event times. Broader event discovery and conditional
prediction remain subsequent milestones. The current continuous stick-slip
least-squares experiments provide classical baselines; they do not train a PINN.

Relevant starting points include [PINN optimization failure modes](https://arxiv.org/abs/2109.01050)
and [the generalization tradeoffs of domain decomposition](https://doi.org/10.1137/21M1447039).
These motivate comparisons; they do not establish novelty or performance of the
proposed hybrid framework.
