"""Fixed grid inference, design baselines, and evaluator for the research pilot.

Controller formulas receive scalar diagnostics only, never truth or future data.
This is trusted numerical code, not a sandbox for arbitrary generated programs.
"""

import time
import numpy as np
from scipy.special import logsumexp

from packages.dynamics.bouncing_ball import BouncingBall
from packages.research.proposals import score


INPUTS = np.array([-4., -2., 0., 2., 4.])
SCALE = np.array([0.25, 0.35])
BASELINES = ("random", "fixed", "local_trace", "local_d", "mean_d", "eig")


def best_candidate(scores):
    """Shared deterministic tie rule for both diagnostics and executed choices."""
    return int(np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-10, rtol=1e-10))[0])


def grid(size):
    z, e = np.meshgrid((np.arange(size) + .5) / size * .25,
                       .6 + (np.arange(size) + .5) / size * .35, indexing="ij")
    return np.column_stack((z.ravel(), e.ravel()))


def prediction(theta, u, t, h0=1.):
    """One scalar sample for each parameter hypothesis, zero or one impact."""
    z, e = theta.T
    a = 9.81 - u
    tau = np.sqrt(2 * (h0 - z) / a)
    dt = t - tau
    return np.where(t < tau, h0 - .5*a*t*t, z + e*a*tau*dt - .5*a*dt*dt)


def sensitivity(theta, u, t):
    z, e = theta.T
    a = 9.81-u
    tau = np.sqrt(2*(1-z)/a)
    # Post-impact one-sided convention at a coincident event; report coincidences.
    post = t >= tau
    return np.column_stack((np.where(post, (1+e)*(2-t/tau), 0.),
                            np.where(post, a*tau*(t-tau), 0.)))


def fisher(theta, u, t, sigma):
    j = sensitivity(theta, u, t) * SCALE / sigma
    return j[:, :, None] * j[:, None, :]


def posterior(theta, observations, sigma):
    logw = np.zeros(len(theta))
    for u, t, y in observations:
        logw -= .5*((prediction(theta, u, t)-y)/sigma)**2
    return np.exp(logw-logsumexp(logw))


def information_gain(means, weights, sigma, refinement=1):
    """Scalar Gaussian-mixture entropy quadrature, including observation noise."""
    if np.ptp(means) < 1e-14:
        return 0.
    tail = 8. if refinement == 1 else 10.
    lo, hi = means.min()-tail*sigma, means.max()+tail*sigma
    count = int(np.ceil((hi-lo)/(sigma/(8*refinement)))) + 1
    y = np.linspace(lo, hi, count)
    density = np.empty(count)
    # Bounded temporary memory, including during convergence checks.
    for start in range(0, count, 128):
        delta = (y[start:start+128, None]-means[None, :])/sigma
        density[start:start+128] = np.exp(-.5*delta**2) @ weights / (sigma*np.sqrt(2*np.pi))
    entropy = np.trapezoid(-density*np.log(np.maximum(density, 1e-300)), y)
    value = float(entropy - .5*np.log(2*np.pi*np.e*sigma*sigma))
    if value < -1e-7 or not np.isfinite(value):
        raise ValueError("invalid EIG quadrature")
    return max(0., value)


def candidate_scores(theta, weights, observations, sigma, t, policy, formula, refinement=1):
    maxw = weights.max()
    modes = np.flatnonzero(weights == maxw)
    distance = np.sum(((theta[modes]-np.array([.125, .775]))/SCALE)**2, axis=1)
    mode = modes[np.argmin(distance)]
    history = sum((fisher(theta, u, ts, sigma) for u, ts, _ in observations),
                  np.zeros((len(theta), 2, 2)))
    local_precision = 12*np.eye(2) + history[mode]
    mean_precision = 12*np.eye(2) + np.einsum("n,nij->ij", weights, history)
    values, diagnostics = [], []
    for u in INPUTS:
        means = prediction(theta, u, t)
        fi = fisher(theta, u, t, sigma)
        mean_fi = np.einsum("n,nij->ij", weights, fi)
        tau = np.sqrt(2*(1-theta[:, 0])/(9.81-u))
        contact = float(weights @ (tau <= t))
        contact = float(np.clip(contact, 0, 1))
        entropy = -sum(p*np.log(p) for p in (contact, 1-contact) if p > 0)
        features = {
            "measurement_snr": float(weights @ (means-weights@means)**2 / sigma**2),
            "contact_entropy": float(entropy),
            "mean_fisher_trace": float(np.trace(mean_fi)),
            "local_trace": float(np.sum(np.diag(fi[mode])/np.diag(local_precision))),
        }
        if policy == "local_trace":
            value = features["local_trace"]
        elif policy in ("local_d", "mean_d"):
            precision, candidate = (local_precision, fi[mode]) if policy == "local_d" else (mean_precision, mean_fi)
            value = float(np.linalg.slogdet(precision+candidate)[1]-np.linalg.slogdet(precision)[1])
        elif policy == "eig":
            value = information_gain(means, weights, sigma, refinement)
        elif policy == "proposed":
            value = score(formula, features)
        else:
            value = 0.
        values.append(value)
        diagnostics.append({"u": float(u), "contact_probability": contact,
                            "boundary_grid_points": int(np.sum(np.abs(t-tau) < 1e-12)),
                            "features": features})
    return np.array(values), diagnostics


def posterior_trajectory(theta, weights, u, times, h0):
    """Evaluate the mixture mean with prefix sums instead of a dense time × grid array."""
    z, e = theta.T
    a = 9.81-u
    tau = np.sqrt(2*(h0-z)/a)
    order = np.argsort(tau)
    coefficients = np.column_stack((weights*(z-h0-(e+.5)*a*tau*tau),
                                    weights*(e+1)*a*tau))
    cumulative = np.vstack((np.zeros(2), np.cumsum(coefficients[order], axis=0)))
    indices = np.searchsorted(tau[order], times, side="right")
    return h0-.5*a*times**2 + cumulative[indices, 0] + times*cumulative[indices, 1]


def error_metrics(theta, weights, truth):
    errors = []
    gx, gw = np.polynomial.legendre.leggauss(3)
    for h0 in (.85, 1.15):
        for u in (-2., 2.):
            a = 9.81-u
            boundaries = np.unique(np.concatenate(([0., .62],
                np.sqrt(2*(h0-theta[:, 0])/a), [np.sqrt(2*(h0-truth[0])/a)])))
            boundaries = boundaries[(boundaries >= 0) & (boundaries <= .62)]
            left, right = boundaries[:-1], boundaries[1:]
            nodes = ((left+right)[:, None]/2 + (right-left)[:, None]/2*gx).ravel()
            quadrature_weights = ((right-left)[:, None]/2*gw).ravel()
            estimate = posterior_trajectory(theta, weights, u, nodes, h0)
            # Independent scalar reference for truth.
            ball = BouncingBall(initial_height=h0-truth[0], restitution=truth[1], applied_acceleration=u)
            actual = np.array([truth[0]+ball.state_at(float(t)).height for t in nodes])
            errors.append(float(np.sqrt(quadrature_weights @ (estimate-actual)**2 / .62)))
    return {"mean_heldout_rmse_m": float(np.mean(errors)),
            "heldout_rmse_m": errors,
            "normalized_parameter_error": ((weights@theta-truth)/SCALE).tolist()}


def world_data(seed, world, regime, sigma):
    truth_rng = np.random.default_rng(np.random.SeedSequence([seed, world, 0]))
    truth = np.array([truth_rng.uniform(0, .25), truth_rng.uniform(.6, .95)])
    initial_rng = np.random.default_rng(np.random.SeedSequence([seed, world, 1]))
    ball = BouncingBall(initial_height=1-truth[0], restitution=truth[1])
    observations = [(0., t, truth[0]+ball.state_at(t).height+sigma*initial_rng.normal())
                    for t in (.1, .2, .42)]
    noise = np.random.default_rng(np.random.SeedSequence([seed, world, regime, 2])).normal(size=(5, 5))*sigma
    return truth, observations, noise


def run_case(config, proposal, world, regime, size=None, refinement=1):
    theta = grid(size or config["grid_size"])
    sigma = proposal["noise_std_m"]
    t = proposal["measurement_times_s"][regime]
    # Regime stream follows the physical time, so changing proposal subsets is paired.
    regime_key = int(round(t*100))
    truth, initial, noise = world_data(config["seed"], world, regime_key, sigma)
    result = {"world": world, "measurement_time_s": t, "grid_size": size or config["grid_size"], "policies": {}}
    for policy in (*BASELINES, "proposed"):
        observations = list(initial)
        weights = posterior(theta, observations, sigma)
        rounds = [{"round": 0, **error_metrics(theta, weights, truth)}]
        random = np.random.default_rng(np.random.SeedSequence([config["seed"], world, regime_key, 3]))
        for step in range(5):
            started = time.perf_counter()
            scores, diagnostics = candidate_scores(theta, weights, observations, sigma, t, policy, proposal["formula"], refinement)
            if policy == "random":
                chosen = int(random.integers(5))
            elif policy == "fixed":
                chosen = step
            else:
                chosen = best_candidate(scores)
            selection_s = time.perf_counter()-started
            u = float(INPUTS[chosen])
            ball = BouncingBall(initial_height=1-truth[0], restitution=truth[1], applied_acceleration=u)
            measured = truth[0]+ball.state_at(t).height+noise[step, chosen]
            observations.append((u, t, float(measured)))
            started = time.perf_counter()
            weights = posterior(theta, observations, sigma)
            inference_s = time.perf_counter()-started
            ordered = np.sort(scores)
            rounds.append({"round": step+1, "u": u, "measurement_m": float(measured),
                           "scores": scores.tolist(), "score_margin": float(ordered[-1]-ordered[-2]),
                           "diagnostics": diagnostics, "selection_s": selection_s, "inference_s": inference_s,
                           "posterior_effective_points": float(1/(weights@weights)),
                           **error_metrics(theta, weights, truth)})
        result["policies"][policy] = rounds
    return result


def numerical_checks(config, proposal):
    theta = grid(config["grid_size"])
    errors = []
    for u in INPUTS:
        for t in (.2, .38, .46, .6):
            tau = np.sqrt(2*(1-theta[:, 0])/(9.81-u))
            away = theta[np.abs(t-tau) > 1e-4]
            for j in range(2):
                shift = np.zeros(2); shift[j] = 1e-6
                fd = (prediction(away+shift, u, t)-prediction(away-shift, u, t))/2e-6
                errors.append(float(np.max(np.abs(fd-sensitivity(away, u, t)[:, j]))))
    prior = posterior(theta, [(0., .1, .95)], .02)
    assert np.max(np.abs(prior-1/len(theta))) < 1e-12
    assert max(errors) < 1e-7
    # Compare grid and sensor integration refinement before inspecting outcomes.
    refinements = []
    stable = True
    for regime, t in enumerate(proposal["measurement_times_s"]):
        for world in range(config["world_count"]):
            _, observations, _ = world_data(config["seed"], world, int(round(t*100)), proposal["noise_std_m"])
            for policy in ("local_trace", "local_d", "mean_d", "eig", "proposed"):
                all_scores = []
                for n, refinement in ((config["grid_size"], 1), (2*config["grid_size"]-1, 2)):
                    th = grid(n); w = posterior(th, observations, proposal["noise_std_m"])
                    scores, _ = candidate_scores(th, w, observations, proposal["noise_std_m"], t, policy, proposal["formula"], refinement)
                    all_scores.append(scores)
                winner = best_candidate(all_scores[0])
                changed = winner != best_candidate(all_scores[1])
                stable = stable and not changed
                refinements.append({"world": world, "time_s": t, "policy": policy,
                                    "coarse_choice": winner, "refined_choice": best_candidate(all_scores[1]),
                                    "max_score_change": float(np.max(np.abs(all_scores[0]-all_scores[1])))})
    return {"derivative_max_absolute_error": max(errors), "preimpact_posterior_uniform": True,
            "identical_prediction_eig": information_gain(np.zeros(3), np.ones(3)/3, .02),
            "initial_choice_grid_refinement_stable": stable, "refinement": refinements,
            "scope": "Initial decisions only; full trajectory refinement required before interpreting superiority."}
