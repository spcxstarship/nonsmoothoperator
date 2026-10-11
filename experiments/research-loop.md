# Small research loop

This is a working, bounded **development** loop:

```text
controller proposal -> validate -> freeze code/inputs -> numerical checks
 -> paired experiments -> refinement audit -> report -> next controller prompt
```

The controller can be an LLM in an interactive session or an explicitly
configured external program. It chooses a hypothesis, measurement regimes,
noise level, and a formula. The runner evaluates it against six numerical
baselines and provides evidence for the next proposal. No provider account or
paid model is connected by default.

## Start and reproduce

Use the existing scientific environment, or install the smaller dependency set
with Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
```

Create a new directory for every campaign:

```sh
.venv/bin/python experiments/research_loop.py init data/research/my-first-loop
.venv/bin/python experiments/research_loop.py step data/research/my-first-loop \
  --proposal experiments/research-proposal.example.json
.venv/bin/python experiments/research_loop.py status data/research/my-first-loop
```

`init` freezes source files, numerical settings, budgets and environment versions.
For a resolution study or fresh development data, set the bounded options at
creation time; existing campaign settings remain immutable:

```sh
.venv/bin/python experiments/research_loop.py init data/research/resolution-study \
  --grid-size 161 --seed 20261008
```

Allowed base grid sizes are 41, 81, and 161; the refinement uses `2*N - 1`.
The seed must be an unsigned 32-bit integer. A fresh seed changes the synthetic
worlds and noise streams. Reusing the seed permits paired resolution comparisons.

`step` validates a proposal, runs the complete batch, and produces:

- `next-prompt.json`: instructions, allowed formula features, remaining budget,
  and previous proposals/results for the next controller decision.
- `proposals/<id>/proposal.json` and `prompt.json`: frozen investigation and context.
- `job-*.json`: checks and per-case outputs, including choices, scores,
  measurements, errors, computation times and a finer-grid rerun.
- `summary.json` and `summary.md`: aggregated evidence and validity flags.
- `manifest.json`: source/artifact hashes, job status, time used and provenance.

To check computational repeatability, copy the first proposal into a local JSON
file, give it a new `id`, set `replicates` to the original ID, and submit another
`step`. Keep its formula, times, and noise unchanged. The runner compares every
scientific field, excluding timing. This reuses the same data; independent
scientific replication requires fresh frozen campaigns with new seed protocols.

Existing campaign directories and proposal IDs are never overwritten. If the
working source changed, invoke the campaign's own
`source/experiments/research_loop.py` using its recorded Python/dependency
versions. Source snapshots capture uncommitted code too. Numerical equivalence
across different hardware is not guaranteed.

## What the LLM may generate

The exact response format is the
[example proposal](research-proposal.example.json). It includes a hypothesis,
mechanism, disconfirming observation, and executable experiment settings.

The initial formula language is deliberately small:

```text
score = sum(weight * transform(feature))
```

One to four terms; finite weights in [-10, 10]; transforms are `identity`,
`log1p`, and `sqrt`. Available features are posterior measurement variance
divided by sensor variance, contact/no-contact entropy, expected Fisher trace,
and local variance-weighted Fisher trace. For example, the first proposal uses

```text
log1p(measurement_snr) + 0.25 * contact_entropy
```

The model may choose sampling times from {0.38, 0.46, 0.60} seconds and sensor
noise from {0.01, 0.02, 0.05} metres. It cannot change the ball equations,
estimator, evaluator, baselines, budget, or hidden data through this interface.
Generated text is never passed to `eval`, a shell, or a Python interpreter.
Arbitrary code generation and derivation checking are later extensions.

Each formula sees only declared diagnostics. Simulator truth and future noise
stay outside the scoring interface. This is a constrained formula evaluator;
it is not an operating-system sandbox for hostile code. The research controller
is separate from the candidate-ID LLM selector in `apps/experiment-design/`.

## Automatic controller transport

An external controller reads **one JSON prompt from stdin**, writes **one JSON
proposal to stdout**, then exits. A new process is used for each decision;
history comes from the recorded prompt. Its executable must be configured by
the operator, never supplied inside a model response.

```sh
.venv/bin/python experiments/research_loop.py run data/research/my-first-loop \
  --controller-command-json '["/absolute/path/to/your-controller"]'
```

The runner automatically calls that controller again after receiving experiment
results. `action: "stop"` ends the loop; use a fresh proposal ID for the stop
response. The controller must return the complete schema even when stopping.
Use environment variables for any adapter credentials. Raw responses and stderr
are retained locally; model/token/spend metadata are not yet collected by this
provider-independent transport. A paid adapter needs its own explicit spend
limit before unattended use. No external model service was called in the first
local campaigns: the assistant in this session authored the two proposals.

The automated transport is integration-tested with a deterministic controller
fixture that receives results and then stops. That fixture is not counted as an
LLM experiment.

## Fixed numerical experiment

The [comparison note](../docs/contact-design-comparison.md) gives the equations,
baseline adaptations and scientific rationale. Each measurement regime uses
four shared prior-drawn worlds, three initial position measurements and five
additional experiments. Policies are random, fixed cycle, local weighted trace,
local log-determinant, posterior-averaged log-determinant, EIG, and the proposed
formula. All use the same grid inference and paired noise streams.

This implementation extends protocol v0.1 with a full adaptive rerun on a finer
grid (by default, 81 × 81 after 41 × 41), also refining EIG quadrature. A changed choice
or any individual held-out trajectory's RMSE changing by more than 0.5 mm raises a validity
flag. The threshold is a development diagnostic, not a significance test.
The original version 2 audit checked mean RMSE; version 3 checks each trajectory
to prevent cancellation across trajectories. Initial and executed decisions use
the same numerical tie rule.
Mixture predictions are evaluated by quadrature split at every relevant impact.

Selection timings include diagnostic construction for every policy, including
random and fixed policies. These are transparent implementation timings, not
optimized computational lower bounds. Finer-grid validation costs count toward
job wall time. The report remains scientifically inconclusive pending numerical
resolution, independent replication, and a frozen confirmation protocol.

## Limits and recovery

Defaults: one sequential worker, two proposals, 30 worker attempts, 120 seconds
per attempt, two controller calls of at most 60 seconds each, and 30 minutes
total subprocess execution. A full proposal uses 13 worker attempts: one check
and 12 paired cases, each containing coarse and refined runs. Filesystem/report
bookkeeping is outside the subprocess-time budget. Limits are frozen at `init`.

Subprocess cleanup terminates the process group even when its leader has exited,
so a controller cannot leave an ordinary child running after its recorded call.
This remains a trusted adapter interface, not containment for code that deliberately
escapes its process group. Failed attempts consume budget
and remain in the ledger. Incomplete batches receive no aggregate method ranking.
Invalid proposals are retained and rejected before numerical execution. A lock
prevents simultaneous writers. Completed outputs are hashed before further work.

On an abrupt crash, a recorded `running` job blocks further execution. Inspect
the recorded PID and reconcile any remaining process; this version requires a
new campaign rather than automatically retrying an uncertain job. There is no
automatic repair, job resumption, detached scheduler, or background process.
`status` displays active job records while keeping restart checks separate.

## First local evidence

The first campaign (`ball-loop-v1`) ran a formula proposal and exact replay across
12 cases each. All scientific outputs matched. Version 2 added the full adaptive
refinement audit; the earlier source and artifacts were retained.

The version 2 audit found changed choices in two cases: local log-determinant at
0.46 s and the proposed formula at 0.60 s, both for development world 1. The
largest mean-RMSE change was approximately 2.50 mm. This prevents a superiority
claim from the observed small differences. Version 2's full 12-case replay also
matched every scientific field, including the refinement flags. Its 26 worker
attempts completed in approximately 16.4 seconds of subprocess wall time and the
loop stopped at its two-proposal budget. The next scientific task is resolving
the numerical sensitivity, then independent replication.

Generated results are under ignored `data/research/`. They have not been staged
or published. The repository's existing document-rights publication restriction
continues to apply.

### Follow-up with subagents

The [parallel-session findings](research-loop-findings.md) document the resolution
diagnosis, runner fixes, and two completed 161 → 321 grid campaigns. All 24 paired
cases passed choice and per-trajectory RMSE refinement checks. The original
formula's small improvement at one schedule did not replicate on fresh data.
Use these findings, rather than the earlier coarse-grid rankings, when choosing
the next experiment. Contact-probability convergence and broader calibration
remain separate questions.

## Verify the implementation

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Tests cover independent simulator agreement, derivatives, posterior and mixture
calculations, information-gain limits, repeatability, proposal validation,
source tampering, timeout accounting, and controller feedback/stop behavior.
