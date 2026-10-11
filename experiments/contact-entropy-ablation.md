# Contact-entropy ablation

Stage: development. Protocol recorded before the ablation runs, 7 October 2026.

Compare the original score `log1p(measurement_snr) + 0.25 * contact_entropy`
against `log1p(measurement_snr)` using the
[variance-only proposal](research-proposal.variance-only.json).

The comparison reuses both completed version 3 development batches: seeds
20261007 and 20261008, four worlds per seed, and measurement times 0.38, 0.46,
and 0.60 seconds. These are eight independent synthetic worlds, each tested in
three regimes, for 24 paired cases. The previously fresh batch is now development
data for this ablation; this is not a confirmation study.

Hold fixed the physics, candidates, three initial observations, five additional
observations, Gaussian noise of 0.02 m, shared grid estimator, six numerical
baselines, 161 × 161 grid, and 321 × 321 refinement. Freeze two new campaigns
with the existing runner, at most 13 worker attempts each (one numerical check
and twelve cases). Use the existing per-job and campaign time limits.

## Analysis fixed before execution

- Verify the original and ablated campaigns have identical source snapshots and
  numerical configurations for each seed, differing only in the proposed score
  and descriptive proposal fields.
- Verify all six baseline policies have identical scientific outputs, excluding
  timings, across each pair. Preserve all failures and refinement warnings.
- Compare final mean held-out trajectory RMSE after five new experiments. Report
  per-world paired changes and per-regime means; negative ablated-minus-original
  changes mean that removing the bonus helped. Also retain complete error curves.
- Count changed proposed-policy action sequences. At the first changed action,
  verify the preceding observations match, then inspect the score components.
- Accept numerical comparability only when both runs pass the choice and
  individual-trajectory RMSE refinement checks. Do not interpret unstable cases
  as wins or silently remove them from averages.
- Report both improvement and deterioration; do not select a favorable schedule
  or introduce further formula tuning during this batch.

The original formula, local Fisher, and EIG comparisons remain useful context.
This ablation tests the contribution of one feature to this particular formula;
it cannot establish whether contact uncertainty is generally useful.

## Outputs

New campaigns: `data/research/ball-ablation-original` and
`data/research/ball-ablation-independent`. Original comparators:
`data/research/ball-loop-v3-resolution` and
`data/research/ball-loop-v3-independent`.

Paired analysis: `data/research/contact-entropy-ablation-comparison/`. The
[comparison script](summarize_contact_ablation.py) verifies artifact hashes,
source/configuration equality, all baseline outputs, and histories at the first
changed decision. Its exact source is copied into the analysis output directory.

```sh
.venv/bin/python experiments/summarize_contact_ablation.py \
  --pair data/research/ball-loop-v3-resolution data/research/ball-ablation-original \
  --pair data/research/ball-loop-v3-independent data/research/ball-ablation-independent \
  --out data/research/contact-entropy-ablation-comparison
```

Use a fresh output directory when replaying the analysis. Campaigns can be
reproduced with the run guide's `init --grid-size 161 --seed ...` and `step`
commands, supplying the variance-only proposal. No numerical engine changes
were made for this ablation.

## Results

Both batches completed all 13 worker attempts and were explicitly stopped.
Total new worker subprocess execution was approximately 132.8 seconds.
All six baselines reproduced exactly at both grid resolutions; proposed-policy
choices and individual held-out errors also passed refinement in all cases.

Removing the contact term changed **2 of 24 action sequences**. The other 22
cases had identical choices, observations, and prediction errors.

Final mean position RMSE in millimetres; each row averages eight worlds and each
world's metric averages four held-out resets:

| Sample time (s) | With entropy | Without entropy | Difference on removal | Local Fisher D |
| --- | --- | --- | --- | --- |
| 0.38 | 23.9302 | 23.9302 | 0 | 23.9302 |
| 0.46 | 8.7321 | 8.4822 | -0.2500 | 7.9672 |
| 0.60 | 3.3409 | 3.3409 | 0 | 2.7167 |

The two changed cases both occurred at 0.46 s, at the third new experiment:

| Seed / world | With entropy (mm) | Without entropy (mm) | Difference on removal |
| --- | --- | --- | --- |
| 20261007 / 1 | 9.6812 | 10.1296 | +0.4484 (worse) |
| 20261008 / 2 | 10.5790 | 8.1307 | -2.4483 (better) |

In both cases, the bonus changed the chosen acceleration from 0 to +2 m/s².
Identical preceding observations and score diagnostics were verified before
the first different choice. Thus the change is attributable to the removed
feature within this fixed implementation and paired sample. Subsequent histories
diverge, as expected in adaptive experiments.

An independent subagent review confirmed matching scientific source/configuration
records, both changed cases, and that the score difference at each first changed
decision is exactly `0.25 * contact_entropy` within 1e-12.

The bonus is not always zero: at least one candidate has a nonzero bonus at
34 of 40 decisions at 0.38 s and 23 of 40 at 0.46 s in the original runs, but it
usually does not change the winning candidate. At 0.60 s every prior-supported
trajectory has already impacted; contact entropy is zero, explaining exact
equality there.

## Interpretation and next decision

The ablation does not support a reliable benefit from this contact-entropy
bonus. Removing it slightly lowers the pooled 0.46 s error, but the entire
change comes from one improvement and one deterioration across eight worlds.
No significance or general-superiority claim follows. The simpler score still
does not beat local Fisher D on these pooled results.

Keep measurement variance alone as a baseline and retain the entropy version
as an unsuccessful candidate with a documented ablation. Before generating
more formula variants, test whether targeting expected prediction-error
reduction, rather than parameter information or measurement disagreement,
improves the match between the selection objective and the evaluation objective.
That is a proposed next investigation, not a conclusion from this ablation.
