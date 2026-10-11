# Contact experiment design: first decision and pilot

Date: 7 October 2026. Decision: **narrow the claim**.
Status: targeted literature and code inspection complete. The
[selector pilot and small research loop](../experiments/research-loop.md) are now
implemented; full refinement exposed numerical sensitivity. Independent
replication and broader reference tracing remain pending.

## What we should investigate

The candidate contribution is a reproducible characterization of selection
failures under sparse observations of dynamic impacts, including the accuracy
and computation tradeoff of inexpensive remedies. Posterior inference and
expected information gain (EIG) are established tools.

**Development question:** With identical inference, observations, and candidate
inputs, does retaining uncertainty across contact/no-contact outcomes change
which experiment is selected, and does that change improve prediction?

This is narrower than claiming a new contact-aware or Bayesian design algorithm.
An advantage on the analytic ball would justify further investigation; it would
not establish generality or paper-level novelty.

## Close comparison

| Source and location | What is already established | Consequence for our comparison |
| --- | --- | --- |
| Sathyanarayan and Abraham, [RSS 2025 final paper](https://www.roboticsproceedings.org/rss21/p118.pdf), Sections IV–V, Eqs. 8–13, Appendix B | Contact-constrained estimation and input design evaluated at a parameter estimate; Gaussian parameter uncertainty is updated. The examples use soft contact sensing. | Keep contact sensitivity, sensor noise, prior information, and uncertainty weighting. Our sparse-position, rigid-impact experiment is an adaptation. |
| Kim, Zhu, and Dollar, [2024 preprint](https://arxiv.org/html/2409.17470v1), Sections IV-B–D, Algorithms 1–2 | Particle beliefs represent multimodal geometry/physical uncertainty. Candidate actions are selected using approximate EIG for quasi-static tactile interactions. | The proposed uncertainty-aware mechanism is already present. Study dynamic impact ambiguity and approximation cost; do not claim particle/EIG selection as new. |
| Odunlami et al., [2026 preprint](https://arxiv.org/html/2603.29862v1), Assumption 3.1, Proposition 3.2, Definition 3.3 | Hybrid sensitivities include event effects. The stated information model includes flow and post-event measurements, under transversal finite transitions. | Retain event-time sensitivity, but count only measurements our sensor actually supplies. Unknown floor height also changes the guard. |

### Published paper and implementation audit

The final RSS PDF was inspected visually, including the formulation on pages
4–6 and appendices on pages 14–15. In the checked passages, Eqs. 8, 12, 13 and
the soft contact model agree with the [v1 preprint](https://arxiv.org/html/2505.12214v1).
This was a targeted comparison, not a complete version diff.

Equation 13 retains prior precision:

\[
\Sigma^+ = (F + \Sigma^{-1})^{-1}.
\]

The [released shape Fisher code](https://github.com/ialab-yale/contact_aware_active_learning/blob/main/experiments/_info_lib/endeff_block_fisher.py)
(`get_fisher`, `get_outer`, `update_paramvar`) differentiates predicted contact
forces, retains diagonal information, scores its variance-weighted trace, and
provides an accumulated precision update. The
[controller](https://github.com/ialab-yale/contact_aware_active_learning/blob/main/experiments/controllers/controllers.py)
passes parameter estimates and variances into planning. These are static reads
of `main`; pin a commit before reproduction. The robotics code was not executed.

Our adaptations change the contact law, sensor, action space, estimator, and
scalarization where specified. Results must be attributed to these adaptations.

## Smallest useful experiment: development protocol v0.1

This protocol is specified before running this selector comparison. It is a
development pilot, not the final confirmation study. The previous observability
and fitting pilots informed its design.

| Item | Specification |
| --- | --- |
| Physics | Existing analytic ball; initial world height 1 m, initial velocity 0, gravity 9.81 m/s². Unknown floor height z and restitution e. |
| Shared prior | Independent uniforms: z in [0, 0.25] m; e in [0.6, 0.95]. |
| Initial data | One neutral-input trajectory, positions at 0.10, 0.20, 0.42 s. |
| Sensor | Independent Gaussian position noise, standard deviation 0.02 m. No velocity, impact flag, event time, or simulator seed in selector inputs. |
| Candidates | Reset the ball and apply constant acceleration u in {-4, -2, 0, 2, 4} m/s². One position measurement per experiment. Repeated candidates are allowed. |
| Measurement regimes | Separate runs with fixed sampling time T = 0.38, 0.46, or 0.60 s. A selector chooses u, not T. Include the later schedule even if it shows no advantage. |
| Budget | Five additional experiments: eight measured positions including initial data. |
| Development cases | Four parameter draws from the declared prior, crossed with three regimes: 12 paired cases. Master generation seed 20261007; distinct recorded streams for parameters, sensor noise, and random selection. |
| Pairing | Same hidden parameters and initial observations across policies; pre-generated noise indexed by case, round, and candidate. Seeds and unused observations stay in the evaluator. |
| Inference | Shared rectangular midpoint grid, initially 41 × 41; log Gaussian likelihood update using all observations. No resampling. Grid MAP for local sensitivities; ties go to the point nearest the prior center, then increasing z/e. |
| Prediction | Shared posterior predictive mean trajectory, evaluated after each round. This keeps estimation differences out of the policy comparison. |
| Held-out evaluation | Four resets: h0 in {0.85, 1.15} m crossed with u in {-2, 2} m/s², over [0, 0.62] s. Noiseless truth is available only to the evaluator. |
| Primary diagnostic | Mean trajectory RMSE over the four held-out resets, divided by 1 m. Integrate over time with impact-aware quadrature. Report each regime and paired policy differences. |
| Secondary diagnostics | Normalized errors of posterior mean z/e; contact probability per candidate; choices and score margins; posterior uncertainty; selector and inference wall times separately. |

Use these six policies:

1. Uniform random choice.
2. Fixed cycle through increasing u, starting at -4.
3. Local variance-weighted Fisher trace, retaining diagonal accumulated precision.
4. Local full-matrix Fisher log-determinant gain, retaining accumulated precision.
5. Posterior-averaged full-matrix Fisher log-determinant gain.
6. Grid EIG reference, integrating over the noisy future measurement.

Policy 4 is the primary local comparator. Policy 3 connects to the inspected
implementation; policy 5 tests whether averaging local information is sufficient.
Any proposed cheap selector must subsequently be compared against policy 6.

### Exact score definitions for our adaptation

Normalize parameters to their prior ranges using S = diag(0.25, 0.35).
For a design d, let J be the row derivative of predicted height with respect to
(z,e), and F(d,theta) = (J S)^T (J S) / sigma². Historical information is the
sum over actual previous measurements, each counted once.

For local scores, evaluate at the current grid MAP and set
P = 12 I + sum_history F. Here 12 I is an explicit moment-matched precision
regularizer for normalized uniform priors, not the Hessian of their log density.
Policy 3 scores sum_i F_candidate[ii] / P[ii]. Policy 4 scores
logdet(P + F_candidate) - logdet(P).

For policy 5, replace every F in that definition with its expectation under the
current grid posterior. Recompute historical information under the current
belief; do not add a previously accumulated matrix a second time.

Policy 6 scores I(theta; Y | history, d). With constant Gaussian sensor variance,
this equals entropy of the predictive Gaussian mixture minus Gaussian noise
entropy. Integrate the scalar predictive density over its component-mean range
extended by eight noise standard deviations, initially with spacing sigma/8.
Verify convergence by halving spacing and extending the bounds. Identical
component predictions must give zero information within numerical tolerance.

Break policy score ties in increasing candidate order; record all ties and
numerically unresolved rankings. Grid and integration convergence must be
checked before interpreting a winner.

## Derivation and validity checks

For a = g-u > 0 and tau = sqrt(2(h0-z)/a), the existing model gives

\[
h(t) = \begin{cases}
h_0 - a t^2/2, & t < \tau, \\
z + e a\tau(t-\tau) - a(t-\tau)^2/2, & \tau < t < (1+2e)\tau.
\end{cases}
\]

Away from the impact boundary its position sensitivity is

\[
J(t) = \begin{cases}
(0,0), & t < \tau, \\
((1+e)(2-t/\tau),\; a\tau(t-\tau)), & t > \tau.
\end{cases}
\]

The floor derivative includes d tau/dz = -1/(a tau). At t = tau a two-sided
parameter derivative need not exist. Detect boundary coincidences, label them,
and report a declared one-sided convention rather than silently treating a
finite-difference stencil across the kink as a smooth derivative.

An immediate observation follows: a locally pre-impact candidate has zero local
information about z/e, while a belief assigning positive probability to distinct
contact outcomes can predict different measurements and positive EIG. This
conditional observation does **not** establish a better candidate ranking or
lower identification error. The pilot tests those additional steps.

On 7 October, 54 central finite-difference comparisons against the existing
`predict` implementation checked this derivative on both sides of impact,
away from the boundary. Maximum absolute discrepancy was 8.42e-11. The earliest
second impact over the proposed design/evaluation extremes is 0.701275 s,
beyond both measurement and evaluation horizons. These are numerical checks,
not selector results.

## First implementation task and stopping rule

Implemented in `experiments/contact_selection_pilot.py` and
`packages/research/ball_design.py`, with the existing scalar analytic simulator
as an independent reference. The grid inference path permits zero or one
impact. The original fitting validator requires every observation horizon to
extend past every possible first impact; do not weaken that existing pilot's
contract or use its PINN unchanged for these cases.

Before the 12 cases, check posterior normalization, truth isolation, derivative
agreement, zero EIG for identical predictions, and stable scores under grid
refinement to 81 × 81. Bound execution to one worker, 120 seconds per case and
30 minutes total. Save config/source hashes, all outcomes, timings, and failures
under an ignored run directory. Do not silently drop timeout cases.

The first deliverable is one table/plot of error versus experiment count plus
candidate choices and contact probabilities. If policies agree, or local Fisher
matches EIG, retain that result. If differences disappear under numerical
refinement, fix the implementation before scientific interpretation. If an
advantage persists, investigate its mechanism with additional development cases
before freezing a larger confirmation study and the proposed 20% effect target.

Remaining literature work before a novelty claim: trace the particle/EIG
predecessors cited by Kim et al.; examine ASID (the RSS comparison reference),
and search dynamic-impact design studies specifically. This note permits a
bounded diagnostic pilot, not a claim that the literature gap is established.
