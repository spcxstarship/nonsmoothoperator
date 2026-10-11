# Bouncing-ball baseline

## Question

Does an explicit flow/jump representation improve estimation of a bouncing
ball's trajectory and impact compared with one smooth trajectory network?

This is the first small hybrid benchmark proposed in
[the workflow note](../docs/openai-navier-stokes-workflow.md). It follows the
standard bouncing-ball model:
free flight obeys `h' = v`, `v' = -g`; at ground contact, height stays zero and
velocity resets to `v_post = -restitution * v_pre`.

## Reference model

`packages/dynamics/bouncing_ball.py` provides exact states from release through
the first rebound. It exposes both pre-impact and post-impact states at the
same impact time. Fix `gravity = 9.81` and `restitution = 0.8` for the first
comparison. Generate training trajectories from a declared set of initial
heights and evaluate on held-out heights. Keep the first experiment within the
first rebound so repeated impacts and Zeno behaviour do not confound it.

Run the reference checks from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

The checks cover the impact reset and energy conservation during flight. They
do not test a PINN; no learning model has been implemented yet.

## Model comparison to implement next

1. Fit the smooth free-flight segment first to establish the PINN training
   pipeline and ODE residual.
2. Compare one smooth network across the impact with a piecewise flow model
   and an explicit jump loss. Use the same observations, collocation budget,
   optimizer budget, and evaluation trajectories.
3. Report held-out height and velocity error, first-impact time error,
   pre/post-impact velocity error, flow residual away from the impact, and
   sensitivity to observations near the impact. Report results across random
   seeds, including failures.
4. Once this comparison is reliable, connect the controlled actuation input to
   the learned hybrid model. Let an LLM choose the next input from a fixed,
   feasible candidate set. Compare its choice with simple non-LLM rules and
   record the prompt, choice, simulation, and held-out outcome. Keep model
   design and training outside the LLM's role. A preliminary selection test
   using a noisy one-parameter estimator is in
   [the experiment-design README](../apps/experiment-design/README.md).

Before training, record the exact train/evaluation heights, observation times,
noise level, network sizes, loss weights, random seeds, and success criteria in
a machine-readable experiment configuration. Keep generated data and model
artifacts outside Git, as required by the repository's public-release policy.
