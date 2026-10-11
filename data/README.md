# Data

Document each dataset's source, license, preparation steps, and expected location here. Large or generated datasets should stay outside Git.

Experiment rounds are stored locally under `data/runs/<study>/<trial>/<round>/`.
`apps/experiment-design/round_store.py` creates them. Each round has a frozen
`context.json`, a validated `decision.json`, a `result.json` naming the executed
candidate, and a `manifest.json` with timestamps, selector identity, file
hashes, and optional reported usage. The selector can be any model or a
deterministic rule; the record format does not depend on a provider. Unknown
model and token fields are `null`. An optional `instructions.txt` stores the
exact task instructions when the selector uses them. Simulation configuration,
including hidden true parameters, belongs in `result.json` after selection;
give the selector only `context.json` and its task instructions.

The `data/runs/` tree is ignored by Git. Back it up separately, and review any
record before deliberately publishing it. The manifest is an audit aid, not a
substitute for a backup or an independently verified simulator result.

Bounded research campaigns live under `data/research/<campaign>/`, also ignored
by Git. They contain synthetic data, exact source snapshots, controller prompts
and responses, numerical outputs, reports, and a hashed attempt ledger. See the
[research-loop guide](../experiments/research-loop.md) for commands and replay.
These development records are distinct from candidate-ID selector rounds.
