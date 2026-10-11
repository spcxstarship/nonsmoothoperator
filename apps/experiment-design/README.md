# Experiment selection prototype

The LLM has one job: choose the next actuation experiment from a supplied list.
It does not design the PINN, change its loss, run code, or decide whether a
result is valid. This boundary follows the project's experiment-design scope.

For now, a Codex subagent makes the choice within the interactive session.
`selector.py` defines the task prompt and checks that its JSON answer names a
candidate from the supplied set. `simulate_choice.py` validates that answer
and runs the selected constant vertical acceleration in the analytic
bouncing-ball reference model. This path needs no OpenAI API key. It is a
session workflow, not an unattended process.

OpenAI documents [Codex use with ChatGPT plans](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)
and [separate API billing](https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform).
This prototype uses the former; it makes no API request.

## Try the interface

Ask Codex to delegate one selection task to a subagent using the current
context file. A reusable request is:

```text
Spawn one experiment-selection subagent. Read
apps/experiment-design/example-context.json and
apps/experiment-design/selector.py. Return only a JSON object with exactly
candidate_id, hypothesis, and rationale. Choose an ID from the context.
Do not edit code, run the simulator, or claim diagnostics that are absent.
```

Save the subagent's JSON answer as `data/decision.json`, then run the reference
simulation from the repository root:

```sh
python3 apps/experiment-design/simulate_choice.py \
  apps/experiment-design/example-context.json \
  data/decision.json > data/simulation.json
```

The example context contains synthetic candidate actuations and no trained
model diagnostics. Its choice is therefore only a plumbing check, not an
informative next experiment. The simulator currently covers one rebound under
a **constant** vertical acceleration. It is a small test system, not the robot
model or a complete active-learning loop.

## Noisy bouncing-ball selection test

`bouncy_demo.py` gives the selector a slightly more useful decision: estimate
the ball's restitution from noisy post-impact speed measurements. The true
restitution is held by the simulator; the selection context contains one
neutral-input observation, its least-squares estimate and standard error, the
measurement noise level, and three feasible constant inputs. Downward input
increases pre-impact speed, while upward input decreases it. The observation
noise has the same standard deviation for every candidate.

From the repository root, prepare a context and ask a subagent to choose one
candidate using that context and `selector.py` only:

```sh
python3 apps/experiment-design/bouncy_demo.py \
  --true-restitution 0.8 --noise-std 0.3 --seed 7 \
  prepare > data/active_demo_context.json
```

Save the subagent's exact JSON reply to `data/active_demo_decision.json`. Then
run the selected experiment and an equal-noise comparison of all candidates:

```sh
python3 apps/experiment-design/bouncy_demo.py \
  --true-restitution 0.8 --noise-std 0.3 --seed 7 \
  evaluate data/active_demo_context.json data/active_demo_decision.json \
  --seeds 100 > data/active_demo_result.json
```

In the first run, the subagent chose `downward`, citing its higher expected
pre-impact speed. The single-run restitution estimate moved from 0.7827 to
0.8099; its stated standard error fell from 0.0677 to 0.0436. Across 100
paired noise seeds, parameter RMSE was 0.0425 for downward, 0.0465 for neutral,
and 0.0520 for upward. The true simulator value was 0.8. These are synthetic
results from this fixed setup, not evidence that an LLM beats a physics-based
selector: with known equal measurement noise, maximizing impact speed is the
obvious non-LLM rule. The model here is a one-parameter least-squares
estimator, **not a PINN**. This exercise checks the selection interface,
information boundary, and result tracking before adding a learned hybrid
model. The generated files under `data/` are ignored by Git.

## Store repeatable rounds

`round_store.py` saves one round in
`data/runs/<study>/<trial>/<round>/`. It is independent of the selector model:
`--selector` identifies a policy such as `codex-subagent` or
`impact-speed-rule`; `--model-id` and `--prompt-version` are optional. A
heuristic must return the same three-field decision JSON as a model. The result
must name the candidate actually run with a top-level `candidate_id`.

Freeze the context **before** asking the selector. For the bouncing-ball demo:

```sh
python3 apps/experiment-design/round_store.py start \
  --study bouncing-ball --trial codex-seed-7 --round 001 \
  --context data/active_demo_context.json --selector codex-subagent \
  --prompt-version select-v1
```

Give the selector the new round's `context.json` and, if present, its frozen
`instructions.txt`. Save its exact reply to
`data/active_demo_decision.json`, then evaluate using that frozen context:

```sh
python3 apps/experiment-design/bouncy_demo.py \
  --true-restitution 0.8 --noise-std 0.3 --seed 7 evaluate \
  data/runs/bouncing-ball/codex-seed-7/001/context.json \
  data/active_demo_decision.json --seeds 100 \
  > data/active_demo_result.json
python3 apps/experiment-design/round_store.py finish \
  data/runs/bouncing-ball/codex-seed-7/001 \
  --decision data/active_demo_decision.json \
  --result data/active_demo_result.json
```

The recorder checks that the choice belongs to the frozen candidate set and
that the result names the same candidate. It keeps the exact JSON bytes and
SHA-256 hashes. It rejects an existing round ID or a second completion. The
manifest has `usage: null` unless the runtime reports counts. When available,
pass `--usage path/to/usage.json` at completion. That file must contain
`source`, `elapsed_ms`, `input_tokens`, `cached_input_tokens`, `output_tokens`,
and `reasoning_tokens`; unavailable values are `null`. The recorder does not
infer token counts or costs. No database or Docker service is needed.
For runs with a textual selector prompt, `start --instructions path/to/prompt.txt`
also freezes its exact bytes. Keep hidden simulator parameters out of the
selection context; the demo writes them to the after-experiment result.

## Research loop to build

1. Establish a hybrid PINN, a diagnostic validation set, and a separate fixed
   test set reserved for evaluating the selection strategy.
2. Define a feasible actuation candidate set and simple non-LLM selection
   baselines before seeing the results.
3. Summarize completed experiments and carefully defined diagnostics for the
   selector. Training loss alone is not a reliable measure of model weakness.
4. Select one candidate, simulate it, collect observations, and update the PINN.
5. Evaluate on the same fixed test set after each update. Compare improvement
   per experiment and per training budget with the non-LLM baselines.

## What to track per selection round

Keep one local record per round, with separate **before-choice** and
**after-experiment** sections. The before-choice section must be frozen before
the subagent runs so later results cannot leak into its input. The current
prompt version is `selector.py`'s `PROMPT_VERSION`.

| Section | Minimum record |
| --- | --- |
| Identity | Round ID, UTC time, prompt version, subagent task ID, model and reasoning setting if exposed. Use `null` for unavailable fields. |
| Before choice | Exact objective, candidate inputs and limits, completed-experiment history, diagnostic definitions and values from information available at decision time, and a hash of the context file. |
| Subagent result | Raw final JSON, validated candidate ID, short hypothesis and rationale, validation outcome, and every retry or correction. |
| Resource use | Elapsed time; input, cached input, output, and reasoning token counts only when the runtime reports them. Record the source of each count. |
| After experiment | Executed input, simulator/data version, random seed, observation set, PINN checkpoint/configuration, and fixed held-out metrics before and after updating the PINN. |

The `hypothesis` and `rationale` fields are the subagent's **stated reasons**.
They are useful for auditing the decision but are not its hidden internal
reasoning. OpenAI's [reasoning documentation](https://developers.openai.com/api/docs/guides/reasoning)
distinguishes visible responses and token counts from hidden reasoning text.
Do not ask for or attempt to reconstruct a private chain of thought.
Do not estimate token counts from response length; absent usage is `null`, not
zero. For the present in-session subagent workflow, the collaboration result
does not provide a token count. A future Codex CLI batch workflow could use
`codex exec --json` and inspect `turn.completed` events where usage is present;
that would be a separate, explicitly tested capture path. See OpenAI's
[Codex JSONL evaluation example](https://developers.openai.com/blog/eval-skills)
and [usage caveats](https://developers.openai.com/api/docs/guides/agents-api/observability).

Judge the LLM selector primarily by improvement in held-out model performance
per experiment against equal-budget non-LLM selectors. Token use and latency
measure efficiency, not scientific success. Preserve failed and corrected
attempts: a corrected prompt is a new selector attempt, not an invisible edit.

For the first hybrid PINN, diagnostics could include trajectory error on a
validation set, event-time error, jump-condition error, flow residual by mode,
and disagreement among independently trained models. Define each metric and
its sampling set before using it. The final test set should not enter the
subagent's context; otherwise the selector can overfit the very outcome used
to assess it.

The model receives aggregate experiment information only. Do not place private
correspondence, credentials, or unpublished third-party data in the context.
Keep generated prompts, decisions, and simulation outputs outside Git until
they have been reviewed for public release.
