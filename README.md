# NonSmoothOperator

Research monorepo for the final year project on physics-informed neural networks, hybrid dynamics, and LLM-assisted experimental design. The [project specification](docs/project-spec.pdf) is the source of requirements.

## Public repository

This repository is public. Contributions, experiment artifacts, and commit history may be visible to anyone. Before adding material, check that it contains no credentials, personal information, private research data, or third-party content without redistribution rights. Agents and contributors should follow [AGENTS.md](AGENTS.md); `.gitignore` alone does not prevent disclosure of tracked files.

## Repository layout

| Path | Purpose |
| --- | --- |
| `apps/experiment-design/` | Runnable interface for LLM-assisted experiment design, if developed. |
| `packages/dynamics/` | Physical systems, equations, and simulation code. |
| `packages/pinn/` | Physics-informed neural network models and losses. |
| `packages/hybrid-model/` | Methods that combine learned and mechanistic dynamics. |
| `packages/shared/` | Code genuinely reused by multiple components. |
| `experiments/` | Reproducible experiment configurations and run notes. |
| `notebooks/` | Exploratory analysis and visualizations. |
| `tests/` | Automated checks for models and experiments. |
| `data/` | Local datasets; document their sources here. |
| `docs/` | Project specification and research documentation. |

The first learning experiment compares a classical estimator with a flow/jump
PINN on noisy bouncing-ball observations. See the
[parameter-identification pilot](experiments/ball-identification.md) for setup,
run commands, evaluation definitions, and current limitations.

The [autonomous research plan](docs/autonomous-research-plan.md) defines the
candidate scientific claim, prior-work review, experiment gates, and execution
limits for the next research campaign.

The first [research loop](experiments/research-loop.md) is runnable: submit a
structured hypothesis and selection formula, compare it against numerical
baselines, and feed the saved evidence into the next proposal. It supports a
configured external controller and exact replay checks without requiring an LLM
service for the numerical experiments.
