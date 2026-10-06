# NonSmoothOperator

Research monorepo for the final year project on physics-informed neural networks, hybrid dynamics, and LLM-assisted experimental design. The [project specification](docs/project-spec.pdf) is the source of requirements.

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

This is an initial scaffold. Add dependencies, commands, and implementation files as the experimental plan becomes concrete.
