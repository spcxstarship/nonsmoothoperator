# What we can learn from OpenAI's Navier–Stokes workflow

Last reviewed: 2026-10-06

## Scope and status

OpenAI published a proposed proof of finite-time blowup for the three-dimensional incompressible Navier–Stokes equations with a smooth external force. It did **not** discover a new fluid equation or train a physics-informed neural network (PINN) to solve this problem. Its paper constructs a flow that starts at rest, remains smooth until a finite time, and then develops unbounded velocity while its kinetic energy stays bounded. This addresses alternatives C and D in the [Clay problem statement](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf).

The [Clay Mathematics Institute](https://www.claymath.org/news/navier-stokes-announcement/) described the problem as *apparently settled* in September 2026. Its [prize rules](https://www.claymath.org/millennium-problems/rules/) require qualifying publication, at least two years, and general acceptance before it considers a proposed solution. The account of the discovery process below is OpenAI's report, not an independently reproducible record of every agent run.

## Reported research workflow

1. **Explore several precise formulations.** OpenAI says it assigned groups of agents both regularity alternatives (A and B) and blowup alternatives (C and D), as well as related, simpler problems.
2. **Use a related problem as a bridge.** An agent group first produced a result for the inviscid Euler equations. OpenAI then concentrated its effort on Navier–Stokes and supplied that intermediate result to agents working on the harder problem.
3. **Run diverse approaches in parallel and combine promising ideas.** OpenAI reports using a coordinating system with roughly 10,000 concurrent agents in the group that produced the Navier–Stokes result. The agents could read a cached internet source and run code. OpenAI says it used Codex to consolidate useful intermediate ideas and feed them back into later attempts.
4. **Turn a candidate mechanism into an exact construction.** The paper builds a shrinking, inward-spiralling vortex. Oscillatory pulses provide momentum transport that cancels the singular residual around the vortex, leaving a smooth external force even as the velocity grows without bound. The proof must control every derivative of that residual; a plausible simulation or approximate solution would not suffice.
5. **Check the result in another representation.** OpenAI released an analytical paper and a [Lean formalization](https://github.com/openai/NavierStokesAndEuler). It reports approximately 88 hours from launching the first agents to the result and another 17 hours for formalization and verification. These figures describe its reported process, not a recipe we could reproduce at project scale.

Sources: [OpenAI's announcement](https://openai.com/index/navier-stokes-solution/) and [the paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf), especially its introduction and physical description.

## Transfer to NonSmoothOperator

Our project calls for a reliable PINN baseline, then hybrid dynamics, then selective LLM assistance. For the first LLM study, we will limit its role to choosing the next actuation experiment from a defined candidate set. Researchers define and train the hybrid PINN. The transferable part is the sequence of **precise formulation → simpler benchmark → candidate mechanism → independent check**. The vortex construction and the scale of OpenAI's agent system are not directly useful PINN methods.

| OpenAI workflow | Practical version for this project |
| --- | --- |
| Test several mathematically distinct formulations | Compare a smooth trajectory PINN, a piecewise flow/jump PINN, and a complementarity or inclusion formulation on the same system. |
| Solve a related, easier problem first | Establish a smooth ODE baseline before adding impacts, friction, or switching. |
| Consolidate promising intermediate results | Give an LLM a compact summary of previous actuation experiments and ask it to choose the next allowed input signal; record its choice and rationale. |
| Check the exact residual, not just an appealing trajectory | Evaluate flow residuals between events, jump conditions at events, event-time error, and longer-horizon rollout error separately. |
| Verify in another representation | Compare learned trajectories with an independent event-driven simulator and test across initial conditions, inputs, and random seeds. |

### First controlled experiment

Use a bouncing ball as a first hybrid system. Fix a restitution coefficient, generate reference trajectories with an event-driven simulator, and reserve some initial heights for evaluation. Compare:

- a single smooth network for the whole trajectory;
- a network with separate flow segments and an explicit impact/reset loss.

Report trajectory error **and** impact-time error, pre/post-impact velocity error, flow residual away from impacts, and sensitivity to sampling density near impacts. Keep the training budget and data fixed between models. This tests whether explicitly representing the discontinuity helps, rather than merely increasing model capacity or collocation points.

Only after this baseline is stable should an LLM select the next actuation experiment. Evaluate its choices against a fixed candidate set, a simple non-LLM selection rule, and a predeclared held-out benchmark. A choice is useful if it improves held-out performance or reduces the data needed to reach it, not merely because its training loss is lower. Estimating where the PINN struggles requires explicit diagnostics and uncertainty checks; the LLM should not infer this from training loss alone.

## Limits and attribution

OpenAI's paper builds on earlier human mathematical work, including work by Diego Córdoba and Luis Martínez-Zoroa. Tristan Buckmaster and Levent Alpöge were pursuing related forced-blowup results and raised questions about the timing and provenance of OpenAI's effort. OpenAI says their private Codex work did not influence its result. These are competing accounts about the research process; this note does not resolve the dispute. See [Buckmaster's statement](https://cims.nyu.edu/~tristanb/statement.pdf) and [OpenAI's response](https://openai.com/index/navier-stokes-solution/).

Finite-time blowup in this Navier–Stokes construction is different from a hybrid system's impact or Zeno behaviour. The lesson for our work is how to structure and verify research, not to treat those phenomena as mathematically equivalent.
