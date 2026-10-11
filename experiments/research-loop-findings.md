# Parallel research session: resolution and fresh-data check

Date: 7 October 2026. Stage: development. Scientific verdict: **the proposed
formula has no replicated advantage in these batches**. This does not reject
every uncertainty-aware design method or establish general superiority of a
local method.

## Work performed

Three subagents independently investigated grid sensitivity, audited numerical
methods, and audited execution safeguards. The coordinating agent integrated
the findings, fixed the runner, and executed two sequential frozen campaigns.
Subagents did not edit implementation files or launch external model services.

Both campaigns used the original formula
`log1p(measurement_snr) + 0.25 * contact_entropy`, all six numerical baselines,
four prior-drawn parameter/noise worlds, three measurement regimes, and five
additional observations. The base grid was 161 × 161 with 321 × 321 refinement.
Campaign seeds were frozen at initialization:

| Campaign | Seed | Purpose | Cases | Worker attempts |
| --- | --- | --- | --- | --- |
| `ball-loop-v3-resolution` | 20261007 | Resolve the existing development cases | 12 | 13 |
| `ball-loop-v3-independent` | 20261008 | Evaluate four fresh synthetic worlds/noise streams | 12 | 13 |

The 24 paired cases represent eight independently generated worlds, each tested
at three schedules. They are not 24 independent physical scenarios. Both
campaigns were explicitly stopped after one proposal; no jobs remain running.
Worker subprocess execution totalled approximately 141.4 seconds, excluding
subagent/model time and separate diagnostic work.

## Numerical diagnosis and fixes

At 41 × 41, the local MAP estimate fell on the wrong side of an impact boundary
for a candidate experiment. The proposed formula also underestimated measurement
variance near a posterior concentrated toward the restitution bound. Increasing
to 81 × 81 resolved those particular choices but exposed changes in
posterior-averaged Fisher and EIG selections. For EIG, refining the measurement
integration at a fixed parameter grid changed scores by less than 9e-16 nats;
parameter discretization caused the choice change.

Full 161 → 321 refinement passed for **every policy in all 24 cases**. All five
choices stayed identical, and the largest individual held-out trajectory RMSE
changes were 0.00704 mm in the original batch and 0.00425 mm in the fresh batch.
These checks establish stability for the recorded choices and errors, not
convergence of every reported contact probability or behavior outside these cases.

The independent numerical audit also checked mixture entropy against adaptive
quadrature, mixture trajectory prediction against direct weighted simulation,
and impact-split RMSE against dense time integration. No core formula defect was
found. The implementation now uses the same tie rule in diagnostics and execution,
and checks each held-out trajectory's RMSE to avoid cancellation in a mean.

The runner audit reproduced a controller child process surviving its parent's
exit. Process-group cleanup now runs even after the parent exits. `status` can
inspect active work without permitting unsafe restart. Grid and seed choices are
bounded and frozen at campaign creation. The full suite passes **27 tests**,
including regressions for cleanup and active-status checks.

## Prediction results

Mean held-out position RMSE after five additional experiments, in millimetres.
Each row averages four worlds; each world metric averages four held-out resets.
These are descriptive development results, without a confirmation interval.

| Batch | Sample time (s) | Local Fisher D | Proposed formula | EIG | Proposed versus local |
| --- | --- | --- | --- | --- | --- |
| Original worlds | 0.38 | 31.7540 | 31.7540 | 31.7540 | Equal |
| Original worlds | 0.46 | 8.2615 | 7.7125 | 7.8247 | 6.64% lower error |
| Original worlds | 0.60 | 3.0855 | 3.7188 | 3.7188 | 20.52% higher error |
| Fresh worlds | 0.38 | 16.1064 | 16.1064 | 16.1064 | Equal |
| Fresh worlds | 0.46 | 7.6728 | 9.7517 | 9.1397 | 27.09% higher error |
| Fresh worlds | 0.60 | 2.3478 | 2.9630 | 3.3385 | 26.20% higher error |

The apparent improvement at 0.46 seconds did not replicate. EIG also does not
consistently beat the local baseline on this prediction-error objective. Optimizing
parameter information and minimizing held-out prediction error are different
objectives; these runs alone do not establish why the rankings differ.

## Evidence and next decision

Exact inputs, snapshots, per-round outputs, all seven policies, timings, and
integrity hashes remain in ignored local artifacts:

- `data/research/ball-loop-v3-resolution/manifest.json`
- `data/research/ball-loop-v3-independent/manifest.json`
- Each campaign's `proposals/measurement-and-contact/summary.json` and `.md`
- `data/research/diagnostics-grid/` for the subagent's sensitivity diagnostics

Earlier campaigns and their numerical warnings remain intact. No files were
staged, committed, or published. See the [run guide](research-loop.md) to reproduce
the campaigns with their frozen snapshots.

Next discriminating experiment: ablate the contact-entropy term against
measurement variance alone, keeping this validated resolution and equal data
budgets. Use development data to understand the mechanism before selecting a new
formula or freezing confirmation. Preserve this unsuccessful proposal in the
research history; do not promote a single favorable schedule into a general claim.

**Follow-up completed:** the [contact-entropy ablation](contact-entropy-ablation.md)
changed only two action sequences out of 24. Removing the bonus helped one case
and hurt another; the simpler score still did not beat local Fisher D in the
pooled results. All numerical and baseline-replay checks passed.
