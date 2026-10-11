# Experiments

Keep each experiment's configuration, random seed, dataset reference, run command, and result summary together so runs can be reproduced. Store large generated outputs outside Git.

Start with the [bouncing-ball baseline](bouncing-ball-baseline.md), which defines
the first reference system, model comparison, and evaluation measures.

The first implemented learning pilot is
[ball parameter identification](ball-identification.md): a classical inverse fit
and a flow/jump PINN trained on the same sparse, noisy position observations.

The numerical selection experiment is specified in the
[contact-design comparison and development protocol](../docs/contact-design-comparison.md).
It compares six numerical selectors with shared grid inference and hidden impact
times. The [small research loop](research-loop.md) now runs those baselines and
one controller-proposed formula with source snapshots, budgets, refinement
checks, reports and repeatability comparisons.

The [contact-entropy ablation](contact-entropy-ablation.md) compares the proposed
score with measurement variance alone on the same eight development worlds.

The [Bouc-Wen input screen](bouc_wen_pilot.py) is a numerical feasibility gate
for experiment selection with an unobserved hysteretic memory state. It fits
four continuous physical parameters from noisy positions and compares three
additional loading profiles with equal duration, measurement count and force
RMS. This dimensionless normalized model is an adaptation of the
[hysteretic benchmark](https://doi.org/10.4121/12967592), not a reproduction of
its published parameters or a test of LLM superiority. The model family,
hysteresis exponent and initial state are known.

Run `.venv/bin/python experiments/bouc_wen_pilot.py --out data/research/NEW`
with a fresh output directory. The runner freezes its protocol and source
snapshots before generating data, checks numerical agreement and noiseless
recovery, and records every completed case under a ten-minute numerical budget.
Failures stop the cohort without an aggregate ranking. The output contains
the protocol, observations, multistart fits, held-out evaluations and results.
All candidate outcomes are evaluated for development diagnostics; no adaptive
selector or LLM is exercised by this script.


The [session-LLM comparison](bouc_wen_selection.py) follows this screen with
eight fresh systems and one acquisition per system. It compares the current
assistant's recorded choices with local Fisher D, local predicted-variance
reduction, fixed mixed frequencies and random choice. All use 128 measured
positions, paired action-indexed noise and the same continuous fitter.

Run the script's `prepare`, `context`, `record` and `evaluate` commands in that
order, each with `--out data/research/NEW`. `record` accepts the actual LLM's
JSON response on standard input. Read only the generated `context.json` while
selecting; evaluator files contain hidden truth and future observations.
Decisions are immutable before evaluation. Source and input hashes are checked
between stages, and numerical choices must survive derivative refinement.

This first comparison uses the existing assistant conversation, including its
development knowledge; it does not invoke fresh independent model instances.
Deployment input definitions are known to both LLM and numerical selectors,
while true responses are withheld until evaluation. This is a procedural
separation, not a filesystem security boundary. The recorded model identity and
token usage are unavailable; results must be described as a session-LLM pilot.
