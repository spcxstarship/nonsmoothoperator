"""Continuous bounded identification using the event-driven contact simulator.

Fit diagnostics are descriptive likelihood/profile diagnostics, not a calibrated
Bayesian posterior. Reserved profiles are explicitly rejected from fit histories.
"""
import importlib.util
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import signal
import sys
import time
import traceback

import numpy as np
from scipy.optimize import least_squares, minimize

_spec = importlib.util.spec_from_file_location(
    'continuous_contact_physics', Path(__file__).with_name('stick_slip_offgrid.py'))
physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physics)

PARAMETERS = ('friction', 'restitution', 'spring', 'damping')
LOWER = np.array([.16, .72, 2.00, .14])
UPPER = np.array([.22, .82, 2.40, .22])
# Deterministic interior Latin-hypercube-style starts; no truth or midpoint seed.
STARTS = np.array([[.125, .625, .875, .375], [.375, .125, .625, .875],
                   [.625, .875, .375, .125], [.875, .375, .125, .625]])
DEFAULT_MAX_NFEV = 65
CLASSES = physics.CLASSES
PROFILES = physics.PROFILES


def parameters(label, normalized):
    values = LOWER + (UPPER - LOWER) * np.asarray(normalized)
    return {'class': label, **dict(zip(PARAMETERS, map(float, values)))}


def starts(label):
    if label not in CLASSES:
        raise ValueError('Unknown declared class')
    return [row.copy() for row in STARTS]


def validate_history(history):
    if not history:
        raise ValueError('History must contain observed positions')
    allowed = {'baseline': physics.BASELINE, **PROFILES}
    for observation in history:
        if observation['candidate_id'] not in allowed:
            raise ValueError('Only baseline and acquisition profiles may be fitted')
        values = np.asarray(observation['positions'], dtype=float)
        if values.shape != physics.TIMES.shape or not np.all(np.isfinite(values)):
            raise ValueError('Each observation must contain 16 finite positions')
    return allowed


def fit_class(history, label, max_nfev=DEFAULT_MAX_NFEV, initial_points=None):
    """Fit one class; retain every local result and explicit optimization failures.

    Standardized Gaussian residuals and bounded normalized coordinates are used.
    max_nfev is SciPy's budget; residual_calls also counts finite differences.
    Linear loss preserves the declared Gaussian observation model.
    """
    if label not in CLASSES:
        raise ValueError('Unknown declared class')
    profiles = validate_history(history)
    rows = []
    begin = time.monotonic()
    initial_points = starts(label) if initial_points is None else initial_points
    for start_id, initial in enumerate(initial_points):
        started = time.monotonic()
        calls = 0
        simulator_calls = 0

        def residual(z):
            nonlocal calls, simulator_calls
            calls += 1
            p = parameters(label, z)
            cache = {}
            pieces = []
            for observation in history:
                name = observation['candidate_id']
                if name not in cache:
                    simulator_calls += 1
                    cache[name] = physics.simulate(p, profiles[name])[0]
                pieces.append((cache[name] - observation['positions']) / physics.NOISE)
            return np.concatenate(pieces)

        row = {'start_id': start_id, 'initial_parameters': parameters(label, initial)}
        try:
            result = least_squares(residual, initial, bounds=(np.zeros(4), np.ones(4)),
                                   method='trf', loss='linear', max_nfev=max_nfev,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                   diff_step=1e-5)
            raw = result.fun * physics.NOISE
            _, singular, directions = np.linalg.svd(result.jac, full_matrices=False)
            cutoff = max(float(singular[0]) * 1e-6, 1e-8)
            row.update(parameters=parameters(label, result.x),
                       normalized_parameters=result.x.tolist(),
                       raw_residuals_m=raw.tolist(), sse_m2=float(raw @ raw),
                       standardized_sse=float(result.fun @ result.fun),
                       optimizer_success=bool(result.success), status=int(result.status),
                       message=str(result.message), nfev=int(result.nfev),
                       njev=int(result.njev or 0), optimality=float(result.optimality),
                       active_bounds=result.active_mask.tolist(),
                       jacobian_singular_values=singular.tolist(),
                       jacobian_right_directions=directions.tolist(),
                       local_jacobian_rank=int(np.sum(singular > cutoff)))
        except TimeoutError:
            raise
        except Exception as exc:
            row.update(optimizer_success=False, status=-99, error=type(exc).__name__,
                       message=str(exc), traceback=traceback.format_exc())
        row.update(residual_calls=calls, simulator_calls=simulator_calls,
                   elapsed_s=time.monotonic() - started)
        rows.append(row)
    finite = [r for r in rows if 'sse_m2' in r and np.isfinite(r['sse_m2'])]
    best = min(finite, key=lambda r: r['sse_m2']) if finite else None
    return {'class': label, 'best': best, 'starts': rows,
            'all_starts_failed': best is None,
            'best_converged': bool(best and best['optimizer_success']),
            'converged_starts': sum(r['optimizer_success'] for r in rows),
            'measurement_count': 16 * len(history),
            'elapsed_s': time.monotonic() - begin,
            'residual_calls': sum(r['residual_calls'] for r in rows),
            'simulator_calls': sum(r['simulator_calls'] for r in rows)}


def fit_history(history, max_nfev=DEFAULT_MAX_NFEV):
    return {label: fit_class(history, label, max_nfev) for label in CLASSES}


def fitted_ensemble(history, fit, delta_sse=9.5):
    """Residual-compatible sensitivity support; NOT posterior samples or coverage.

    Box corners preserve unconstrained directions for the all-stuck baseline.
    Local Jacobian directions supply smaller perturbations when identifiable.
    The cutoff is a fixed heuristic radius, not a confidence region assertion.
    No reserved profile is evaluated or consulted here.
    """
    import itertools
    profiles = validate_history(history)
    best = fit['best']
    if best is None:
        return {'members': [], 'failures': [], 'all_starts_failed': True}
    center = np.asarray(best['normalized_parameters'])
    candidates = [center, *STARTS]
    candidates.extend(np.array(v, float) for v in itertools.product([0., 1.], repeat=4))
    candidates.extend(np.array(r['normalized_parameters']) for r in fit['starts']
                      if 'normalized_parameters' in r)
    for singular, direction in zip(best['jacobian_singular_values'],
                                   best['jacobian_right_directions']):
        direction = np.array(direction)
        # sigma-scaled local displacements and larger finite probes both matter
        # because event changes can invalidate a local derivative.
        for radius in sorted(set([min(1., 1. / max(singular, 1e-8)),
                                  min(1., 2. / max(singular, 1e-8)), .25, .5, 1.])):
            for sign in [-1, 1]:
                candidates.append(np.clip(center + sign * radius * direction, 0., 1.))
    unique = {}
    for z in candidates:
        unique.setdefault(tuple(np.round(z, 8)), z)
    members, failures = [], []
    started = time.monotonic()
    simulator_calls = 0
    for z in unique.values():
        p = parameters(fit['class'], z)
        try:
            predicted = {}
            for name in dict.fromkeys(o['candidate_id'] for o in history):
                simulator_calls += 1
                predicted[name] = physics.simulate(p, profiles[name])[0]
            residual = np.concatenate([predicted[o['candidate_id']] - o['positions']
                                       for o in history])
            sse = float(residual @ residual) / physics.NOISE ** 2
            if sse <= best['standardized_sse'] + delta_sse + 1e-8:
                members.append({'parameters': p, 'standardized_sse': sse,
                                'normalized_parameters': z.tolist()})
        except TimeoutError:
            raise
        except Exception as exc:
            failures.append({'parameters': p, 'error': str(exc),
                             'traceback': traceback.format_exc()})
    return {'members': members, 'failures': failures, 'delta_sse_cutoff': delta_sse,
            'proposed_count': len(unique), 'elapsed_s': time.monotonic() - started,
            'simulator_calls': simulator_calls,
            'interpretation': 'Residual-compatible sensitivity support, not posterior samples'}


def calibration_cases():
    """Prespecified diverse synthetic cases distinct from benchmark parameters."""
    values = [(.169, .739, 2.07, .153), (.211, .801, 2.33, .207),
              (.181, .793, 2.11, .199), (.203, .747, 2.29, .161)]
    return [{'class': label, **dict(zip(PARAMETERS, row))}
            for label, row in zip(CLASSES, values)]


def calibration_history(p, noise=False):
    rng = np.random.default_rng(591031)
    history = []
    for name in ['baseline', *physics.FIXED]:
        profile = physics.BASELINE if name == 'baseline' else PROFILES[name]
        y = physics.simulate(p, profile)[0]
        if noise:
            y = y + rng.normal(0, physics.NOISE, len(y))
        history.append({'candidate_id': name, 'positions': y.tolist()})
    return history


SELF = 'experiments/stick_slip_continuous.py'
POLICIES = ['fresh_llm', 'numerical_disagreement', 'fixed_balanced', 'uniform_random']
FIT_CONFIG = {'bounds': dict(zip(PARAMETERS, zip(LOWER.tolist(), UPPER.tolist()))),
              'implementation_version': 'event-refit-repair-v2',
              'starts': STARTS.tolist(), 'max_nfev_per_start': DEFAULT_MAX_NFEV,
              'ensemble_delta_standardized_sse': 9.5, 'loss': 'linear',
              'finite_difference_step': 1e-5, 'optimizer_tolerances': 1e-8,
              'probe_repair': 'At most two bounded Powell(maxfev=400, order=e,k,c,f) then least-squares refits. Failed or worse local repair preserves the directly reverified best observed point, flags unresolved/nonconverged, and retains failure stacks. Remaining improvement >1e-3 after budget also promotes the best observed point. Bounded finite-difference diagnostics try both directions; unavailable derivatives are reported as null, never zero.'}
INSTRUCTIONS = '''Choose ONE next force-profile experiment to improve position prediction and identify physical parameters and the declared model class. Use only observed noisy positions, fitted diagnostics, declared mechanics, and candidate descriptions. Three sequential choices total. Return ONLY JSON with candidate_id, hypothesis, rationale. Candidate must be an exact supplied ID; hypothesis and rationale must be nonempty strings under 700 characters. No changes to sample times, model or measurement count. Relative fit weights and sensitivity support are heuristics, not calibrated posterior probabilities. Do not request hidden measurements, test profiles, or benchmark information.'''
write, read, sha, now = physics.write, physics.read, physics.sha, physics.now


def promote_probe(history, fit, support, probe, reason='bounded repair limit'):
    """Preserve best observed evidence after a bounded unsuccessful local search."""
    started = time.monotonic()
    profiles = validate_history(history)
    simulation_calls = 0
    residual_calls = 0
    def residual(z):
        nonlocal simulation_calls, residual_calls
        residual_calls += 1
        predicted = {}
        for name in dict.fromkeys(o['candidate_id'] for o in history):
            simulation_calls += 1
            predicted[name] = physics.simulate(parameters(fit['class'], z), profiles[name])[0]
        return np.concatenate([predicted[o['candidate_id']] - o['positions'] for o in history]) / physics.NOISE
    fit['pending_probe_promotion'] = {'reason': reason, 'probe': probe}
    z = np.array(probe['normalized_parameters'])
    values = residual(z)
    if not np.all(np.isfinite(values)) or not np.isclose(
            values @ values, probe['standardized_sse'], atol=1e-6, rtol=1e-9):
        raise RuntimeError('Cannot reproduce the retained finite parameter candidate')
    jac = []
    derivative_failures = []
    for i in range(4):
        column = None
        for step in [1e-5, -1e-5]:
            displaced = z.copy()
            displaced[i] += step
            if not 0 <= displaced[i] <= 1:
                continue
            try:
                candidate = (residual(displaced) - values) / step
                if not np.all(np.isfinite(candidate)):
                    raise RuntimeError('Nonfinite diagnostic derivative')
                column = candidate
                break
            except TimeoutError:
                raise
            except Exception as exc:
                derivative_failures.append({'parameter': PARAMETERS[i], 'step': step,
                                            'error': str(exc), 'traceback': traceback.format_exc()})
        jac.append(column)
    available = all(column is not None for column in jac)
    if available:
        jac = np.array(jac).T
        _, singular, directions = np.linalg.svd(jac, full_matrices=False)
        rank = int(np.sum(singular > max(singular[0] * 1e-6, 1e-8)))
        optimality = float(np.max(np.abs(jac.T @ values)))
    else:
        singular, directions, rank, optimality = np.array([]), np.empty((0, 4)), None, None
    raw = values * physics.NOISE
    row = {'start_id': 'promoted_probe', 'parameters': parameters(fit['class'], z),
           'normalized_parameters': z.tolist(), 'raw_residuals_m': raw.tolist(),
           'sse_m2': float(raw @ raw), 'standardized_sse': float(values @ values),
           'optimizer_success': False, 'status': -98,
           'message': 'probe_promoted_unresolved: ' + reason + '; no global optimum claim',
           'nfev': 0, 'njev': 0, 'residual_calls': residual_calls, 'simulator_calls': simulation_calls,
           'optimality': optimality, 'diagnostic_derivatives_available': available,
           'diagnostic_derivative_failures': derivative_failures,
           'active_bounds': [int(-1 if x <= 1e-8 else 1 if x >= 1-1e-8 else 0) for x in z],
           'jacobian_singular_values': singular.tolist(), 'jacobian_right_directions': directions.tolist(),
           'local_jacobian_rank': rank,
           'elapsed_s': time.monotonic() - started}
    fit.pop('pending_probe_promotion', None)
    fit['starts'].append(row)
    fit['best'] = row
    fit['best_converged'] = False
    fit['unresolved_optimization'] = True
    for key in ['elapsed_s', 'residual_calls', 'simulator_calls']:
        fit[key] += row[key]
    support['members'] = [m for m in support['members'] if m['standardized_sse'] <= row['standardized_sse'] + 9.5 + 1e-8]
    if not any(np.array_equal(m['normalized_parameters'], z) for m in support['members']):
        support['members'].append(probe)


class FitAnalysisError(RuntimeError):
    """Retain class diagnostics when no defensible comparison can be made."""
    def __init__(self, message, fits):
        super().__init__(message)
        self.fit_diagnostics = fits


def analyze_history(history):
    fits = fit_history(history)
    if any(f['all_starts_failed'] for f in fits.values()):
        raise FitAnalysisError('At least one class has no finite fit; abort comparison', fits)
    try:
        return analyze_fits(history, fits)
    except TimeoutError:
        raise
    except Exception as exc:
        raise FitAnalysisError(str(exc), fits) from exc


def analyze_fits(history, fits):
    ensembles = {}
    for label, fit in fits.items():
        fit['unresolved_optimization'] = False
        repair_log = []
        fit['ensemble_repairs'] = repair_log
        fit['support_failure_history'] = []
        total_support_calls = 0
        total_support_time = 0.
        for attempt in range(3):
            support = fitted_ensemble(history, fit)
            fit['support_failure_history'].extend(support.get('failures', []))
            total_support_calls += support['simulator_calls']
            total_support_time += support['elapsed_s']
            if not support['members']:
                raise RuntimeError('Empty class sensitivity support')
            probe = min(support['members'], key=lambda row: row['standardized_sse'])
            improvement = fit['best']['standardized_sse'] - probe['standardized_sse']
            if improvement <= 1e-3:
                break
            if attempt == 2:
                promote_probe(history, fit, support, probe)
                break
            # At a stick/release boundary finite differences can straddle a
            # discontinuity. A bounded derivative-free repair can move along it.
            profiles = validate_history(history)
            repair_started = time.monotonic()
            objective_calls = 0
            simulation_calls = 0
            failures = []
            best_probe = {'sse': probe['standardized_sse'], 'z': np.array(probe['normalized_parameters'])}
            def objective(z):
                nonlocal objective_calls, simulation_calls
                objective_calls += 1
                predictions = {}
                try:
                    for name in dict.fromkeys(o['candidate_id'] for o in history):
                        simulation_calls += 1
                        predictions[name] = physics.simulate(parameters(label, z), profiles[name])[0]
                    value = sum(float(np.sum((predictions[o['candidate_id']] - o['positions']) ** 2))
                                for o in history) / physics.NOISE ** 2
                    if value < best_probe['sse']:
                        best_probe.update(sse=value, z=np.array(z).copy())
                    return value
                except TimeoutError:
                    raise
                except Exception as exc:
                    failures.append({'parameters': parameters(label, z), 'error': str(exc),
                                     'traceback': traceback.format_exc()})
                    return 1e20
            polished = minimize(objective, np.array(probe['normalized_parameters']),
                                 method='Powell', bounds=[(0., 1.)] * 4,
                                 options={'maxfev': 400, 'xtol': 1e-8, 'ftol': 1e-8,
                                          'direc': np.eye(4)[[1, 2, 3, 0]]})
            repair_initial = best_probe['z']
            powell_elapsed = time.monotonic() - repair_started
            repair = fit_class(history, label, initial_points=[repair_initial])
            repair['derivative_free_repair'] = {'success': bool(polished.success), 'message': str(polished.message),
                                               'standardized_sse': float(polished.fun), 'objective_calls': objective_calls,
                                               'best_observed_standardized_sse': float(best_probe['sse']),
                                               'simulator_calls': simulation_calls, 'elapsed_s': powell_elapsed,
                                               'failures': failures}
            repair['elapsed_s'] += powell_elapsed
            repair['residual_calls'] += objective_calls
            repair['simulator_calls'] += simulation_calls
            repair_log.append({'probe_improvement_standardized_sse': improvement,
                               'fit': repair})
            for key in ['elapsed_s', 'residual_calls', 'simulator_calls', 'converged_starts']:
                fit[key] += repair[key]
            fit['starts'].extend(repair['starts'])
            if repair['best'] is None or repair['best']['standardized_sse'] > best_probe['sse'] + 1e-3:
                retained = {'parameters': parameters(label, best_probe['z']),
                            'normalized_parameters': best_probe['z'].tolist(),
                            'standardized_sse': float(best_probe['sse'])}
                promote_probe(history, fit, support, retained,
                              reason='local refit failed or lost a verified finite candidate')
                repair_log[-1]['outcome'] = 'verified_probe_retained_unresolved'
                break
            fit['best'] = min(fit['starts'], key=lambda row: row.get('standardized_sse', float('inf')))
            fit['best_converged'] = fit['best']['optimizer_success']
        fit['ensemble_repairs'] = repair_log
        support['total_support_simulator_calls'] = total_support_calls
        support['total_support_elapsed_s'] = total_support_time
        ensembles[label] = support
    if any(not e['members'] for e in ensembles.values()):
        raise RuntimeError('Empty class sensitivity support')
    sse = np.array([fits[label]['best']['standardized_sse'] for label in CLASSES])
    relative = np.exp(-.5 * (sse - sse.min()))
    relative /= relative.sum()
    return {'fits': fits, 'ensembles': ensembles,
            'relative_fit_weights': dict(zip(CLASSES, relative.tolist())),
            'minimum_sse_class': CLASSES[int(np.argmin(sse))],
            'tied_classes': [label for label, value in zip(CLASSES, sse) if value - sse.min() <= 1e-8],
            'competitive_classes': [label for label, value in zip(CLASSES, sse) if value - sse.min() <= 9.5],
            'interpretation': 'Relative optimized Gaussian fit scores; no integration over parameters and no calibrated posterior'}


def cached_analysis(root, history):
    key = hashlib.sha256(json.dumps({'history': history, 'config': FIT_CONFIG},
                                    sort_keys=True).encode()).hexdigest()
    path = root / 'fit-cache' / (key + '.json')
    if not path.exists():
        try:
            analysis = analyze_history(history)
        except TimeoutError:
            raise
        except Exception as exc:
            failure_path = root / 'fit-failures' / (key + '.json')
            if not failure_path.exists():
                write(failure_path, {'cache_key': key, 'history': history,
                                     'error': str(exc), 'traceback': traceback.format_exc(),
                                     'fit_diagnostics': getattr(exc, 'fit_diagnostics', None)})
            raise RuntimeError('Fit analysis failed; diagnostics: ' + str(failure_path)) from exc
        write(path, analysis)
    return read(path), key


def predictive_moments(analysis, profile):
    means, variances, weights = [], [], []
    for label in CLASSES:
        values = np.array([physics.simulate(m['parameters'], profile)[0]
                           for m in analysis['ensembles'][label]['members']])
        means.append(values.mean(axis=0))
        variances.append(values.var(axis=0))
        weights.append(analysis['relative_fit_weights'][label])
    means, variances, weights = np.array(means), np.array(variances), np.array(weights)
    mean = weights @ means
    within = weights @ variances
    between = weights @ ((means - mean) ** 2)
    return mean, within, between


def numerical_decision(analysis):
    scores = {}
    for name, profile in PROFILES.items():
        _, within, between = predictive_moments(analysis, profile)
        scores[name] = {'total': float(np.mean(within + between) / physics.NOISE ** 2),
                        'within_class': float(np.mean(within) / physics.NOISE ** 2),
                        'between_class': float(np.mean(between) / physics.NOISE ** 2)}
    selected = max(PROFILES, key=lambda name: scores[name]['total'])
    return {'candidate_id': selected, 'scores': scores,
            'rationale': 'Maximum total fitted sensitivity-ensemble predictive variance / noise variance; deterministic listed-order ties; heuristic, not expected information gain.'}


def context(history, analysis, round_number):
    base = physics.context(history, {}, round_number)
    base['objective'] = 'Improve position prediction and identify physical parameters and declared contact/friction model.'
    base['known_model'].pop('balanced_parameter_prior')
    base['known_model']['continuous_parameter_bounds'] = FIT_CONFIG['bounds']
    base['known_model']['parameter_domain'] = 'Independent continuous bounded parameters within each of four declared classes.'
    base['diagnostics'] = {
        'noise_std_m': physics.NOISE,
        'relative_fit_weights': analysis['relative_fit_weights'],
        'minimum_sse_class': analysis['minimum_sse_class'],
        'competitive_classes': analysis['competitive_classes'],
        'class_fits': {label: {'parameters': f['best']['parameters'],
                             'standardized_sse': f['best']['standardized_sse'],
                             'sse_per_measurement_m2': f['best']['sse_m2'] / (16 * len(history)),
                             'jacobian_rank': f['best']['local_jacobian_rank'],
                             'active_bounds': f['best']['active_bounds'],
                             'best_converged': f['best_converged'],
                             'unresolved_optimization': f['unresolved_optimization'],
                             'compatible_support_count': len(analysis['ensembles'][label]['members'])}
                       for label, f in analysis['fits'].items()},
        'limitations': 'Bounded multistart local fitting can miss optima. Fit weights and sensitivity support are descriptive, not calibrated posterior or confidence intervals. Rank is local; large residuals can reflect optimizer failure, noise, or model mismatch.'}
    return base


def runtime(root):
    total = sum(read(p)['elapsed_s'] for p in root.glob('stage-*-runtime.json'))
    if total >= 1200:
        raise RuntimeError('1200-second active numerical runtime exhausted')
    return total


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    repository = Path(__file__).resolve().parents[1]
    for relative in [SELF, 'experiments/stick_slip_offgrid.py', 'tests/test_stick_slip_continuous.py']:
        destination = root / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, destination)
    protocol = {'study': 'Continuous parameter identification pilot in four declared mechanisms',
                'frozen_at': now(), 'fit_config': FIT_CONFIG, 'cases': physics.offgrid_cases(),
                'profiles': PROFILES, 'reserved_profiles': physics.RESERVED,
                'times': physics.TIMES.tolist(), 'noise': physics.NOISE,
                'baseline': physics.BASELINE, 'policies': POLICIES, 'fixed_sequence': physics.FIXED,
                'rounds': 3, 'measurement_budget': 64, 'maximum_selector_calls': 24,
                'numerical_runtime_s': 1200, 'source_sha256': sha(root / 'sources' / SELF),
                'source_hashes': physics.inventory(root / 'sources'),
                'selector_instructions': INSTRUCTIONS,
                'numerical_rule': 'Maximum total fitted-ensemble predictive variance averaged over 16 positions divided by noise variance; within- and between-class contributions recorded; listed-order ties.',
                'ensemble_rule': 'Fixed box corners, LHS starts, local optima and clipped Jacobian-direction perturbations; retain points within standardized SSE 9.5 of class minimum; uniform within-class member weights and exp(-delta optimized standardized SSE/2) relative class fit weights; heuristic only.',
                'metrics': 'Primary mean case-level aggregate reserved-position RMSE; per-profile RMSE; minimum-SSE class with ties and competitive classes; parameter error; local Jacobian rank; residuals by acquisition profile; optimizer convergence, bounds, actual evaluations, failures and timing. No Bayesian NLL or confident-wrong statistic.',
                'cache': 'SHA256 of exact observed history plus fit configuration, no warm starts',
                'development_disclosure': 'Engineering pilot: software smoke tests used benchmark synthetic histories/control choices before freeze; failed smoke traces preserved locally, no independent generalization claim. Prospective 24 model calls occur only after reviewed source/protocol freeze.',
                'limitations': 'Synthetic eight-case pilot; two noise replicates per class share parameters. Frozen mechanics, no novel mechanism discovery, no PINN or language-model weight training.'}
    write(root / 'protocol.json', protocol)
    write(root / 'environment.json', {'python': sys.version, 'numpy': np.__version__,
                                      'device': 'cpu', 'started_at': now(), 'usage': None})
    write(root / 'input-manifest.json', {'artifacts': physics.inventory(root)})
    truths = {case['case_id']: {name: physics.simulate(case['truth'], profile)[0].tolist()
                               for name, profile in {'baseline': physics.BASELINE, **PROFILES, **physics.RESERVED}.items()}
              for case in protocol['cases']}
    write(root / 'truth-predictions.json', truths)
    for case in protocol['cases']:
        observed = np.array(truths[case['case_id']]['baseline']) + np.random.default_rng(case['baseline_seed']).normal(0, physics.NOISE, 16)
        for policy in POLICIES:
            write(root / 'cases' / case['case_id'] / policy / 'history-0.json',
                  [{'candidate_id': 'baseline', 'positions': observed.tolist()}])
    write(root / 'prepared-state.json', {'artifacts': {str(p.relative_to(root)): sha(p)
                                                     for p in root.rglob('*.json') if p.name in ['truth-predictions.json', 'history-0.json']}})
    physics.trace(root, 'freeze', 'Source, protocol, seeds and evaluator inputs frozen before policy outcomes.', [root / 'protocol.json'])
    write(root / 'stage-prepare-runtime.json', {'elapsed_s': time.monotonic() - started})
    freeze_round(root, 1)


def freeze_round(root, r):
    runtime(root)
    started = time.monotonic()
    ledger = []
    for case in read(root / 'protocol.json')['cases']:
        for policy in POLICIES:
            route = root / 'cases' / case['case_id'] / policy
            history = read(route / f'history-{r-1}.json')
            try:
                analysis, key = cached_analysis(root, history)
            except TimeoutError:
                raise
            except Exception as exc:
                raise RuntimeError(f'{case["case_id"]}/{policy}, fit after round {r-1}: {exc}') from exc
            write(route / f'fit-{r-1}.json', {'cache_key': key, 'cache_path': f'fit-cache/{key}.json'})
            if policy == 'fresh_llm':
                folder = route / f'selector-{r}'
                write(folder / 'context.json', context(history, analysis, r))
                (folder / 'instructions.txt').write_text(INSTRUCTIONS)
                ledger.append({'case_id': case['case_id'], 'context': str((folder / 'context.json').resolve()),
                               'instructions': str((folder / 'instructions.txt').resolve())})
            else:
                if policy == 'numerical_disagreement':
                    decision = numerical_decision(analysis)
                elif policy == 'fixed_balanced':
                    decision = {'candidate_id': physics.FIXED[r-1], 'rationale': 'Frozen balanced sequence'}
                else:
                    index = int(np.random.default_rng(case['random_seed'] + 100*r).integers(len(PROFILES)))
                    decision = {'candidate_id': list(PROFILES)[index], 'rationale': 'Seeded uniform random'}
                write(route / f'decision-{r}.json', {'policy': policy, **decision})
    write(root / f'round-{r}-contexts.json', ledger)
    write(root / f'round-{r}-frozen.json', {'artifacts': {str(p.relative_to(root)): sha(p)
          for p in root.rglob('*') if p.is_file() and (p.name in [f'history-{r-1}.json', f'fit-{r-1}.json', f'decision-{r}.json'] or f'selector-{r}' in p.parts or 'fit-cache' in p.parts)}})
    write(root / f'stage-freeze-{r}-runtime.json', {'elapsed_s': time.monotonic() - started})
    physics.trace(root, 'decision', f'Round {r} contexts and control decisions frozen before acquisition.', [root / f'round-{r}-frozen.json'])
    print('CONTEXTS=' + str((root / f'round-{r}-contexts.json').resolve()), flush=True)


def acquire(root, r, map_path):
    if (root / f'round-{r}-lock.json').exists():
        raise ValueError('Round already started; no retries')
    if r > 1 and not (root / f'round-{r-1}-complete.json').exists():
        raise ValueError('Previous round incomplete')
    verify(root)
    entries = read(map_path)
    expected = {row['case_id'] for row in read(root / f'round-{r}-contexts.json')}
    if len(entries) != 8 or {row['case_id'] for row in entries} != expected:
        raise ValueError('Require eight unique actual fresh selector decisions')
    for entry in entries:
        d = read(Path(entry['decision']))
        read(Path(entry['metadata']))
        if set(d) != {'candidate_id', 'hypothesis', 'rationale'} or d['candidate_id'] not in PROFILES or any(not isinstance(d[k], str) or not 0 < len(d[k]) < 700 for k in ['hypothesis', 'rationale']):
            raise ValueError('Invalid fresh selector decision')
    write(root / f'round-{r}-lock.json', {'started_at': now()})
    started = time.monotonic()
    shutil.copyfile(map_path, root / f'round-{r}-decision-map.json')
    for entry in entries:
        route = root / 'cases' / entry['case_id'] / 'fresh_llm'
        shutil.copyfile(entry['decision'], route / f'decision-{r}.json')
        shutil.copyfile(entry['metadata'], route / f'metadata-{r}.json')
    truths = read(root / 'truth-predictions.json')
    for case in read(root / 'protocol.json')['cases']:
        for policy in POLICIES:
            route = root / 'cases' / case['case_id'] / policy
            choice = read(route / f'decision-{r}.json')['candidate_id']
            seed = case['candidate_seed_base'] + 100*r + list(PROFILES).index(choice)
            y = np.array(truths[case['case_id']][choice]) + np.random.default_rng(seed).normal(0, physics.NOISE, 16)
            observation = {'candidate_id': choice, 'positions': y.tolist()}
            write(route / f'new-observations-{r}.json', observation)
            write(route / f'history-{r}.json', read(route / f'history-{r-1}.json') + [observation])
            physics.trace(root, 'action', f'{case["case_id"]}/{policy}, round {r}: {choice}; shared candidate/round noise.', [route / f'decision-{r}.json', route / f'history-{r}.json'])
    write(root / f'stage-acquire-{r}-runtime.json', {'elapsed_s': time.monotonic() - started})
    write(root / f'round-{r}-complete.json', {'artifacts': {str(p.relative_to(root)): sha(p) for p in root.rglob(f'history-{r}.json')}})
    if r < 3:
        freeze_round(root, r+1)
    else:
        finish(root)


def stage_metrics(history, analysis, truth, test_truth):
    per_profile, errors = {}, []
    for name, profile in physics.RESERVED.items():
        mean, _, _ = predictive_moments(analysis, profile)
        error = mean - test_truth[name]
        errors.extend(error.tolist())
        per_profile[name] = float(np.sqrt(np.mean(error ** 2)))
    selected = analysis['minimum_sse_class']
    best = analysis['fits'][selected]['best']
    class_diagnostics = {}
    for label, fit in analysis['fits'].items():
        b = fit['best']
        residuals = np.array(b['raw_residuals_m']).reshape(len(history), 16)
        class_diagnostics[label] = {
            'parameters': b['parameters'], 'standardized_sse': b['standardized_sse'],
            'jacobian_rank': b['local_jacobian_rank'], 'active_bounds': b['active_bounds'],
            'best_converged': fit['best_converged'], 'converged_starts': fit['converged_starts'],
            'unresolved_optimization': fit['unresolved_optimization'],
            'residual_calls': fit['residual_calls'], 'simulator_calls': fit['simulator_calls'],
            'fit_elapsed_s': fit['elapsed_s'],
            'residuals_by_observation': [{'candidate_id': o['candidate_id'], 'sse_m2': float(v @ v),
                                         'mean_m': float(v.mean())} for o, v in zip(history, residuals)]}
    parameter_error = {k: best['parameters'][k] - truth[k] for k in PARAMETERS}
    return {'heldout_position_rmse_m': float(np.sqrt(np.mean(np.square(errors)))),
            'reserved_rmse_m': per_profile, 'minimum_sse_class': selected,
            'minimum_sse_class_correct': selected == truth['class'],
            'selected_fit_unresolved': analysis['fits'][selected]['unresolved_optimization'],
            'class_tie': len(analysis['tied_classes']) > 1, 'tied_classes': analysis['tied_classes'],
            'competitive_classes': analysis['competitive_classes'],
            'truth_class_competitive': truth['class'] in analysis['competitive_classes'],
            'relative_fit_weights': analysis['relative_fit_weights'],
            'minimum_sse_per_measurement_m2': best['sse_m2'] / (len(history) * 16),
            'selected_class_parameter_error': parameter_error,
            'normalized_parameter_rmse': float(np.sqrt(np.mean([(parameter_error[k] / (hi-lo)) ** 2 for k, lo, hi in zip(PARAMETERS, LOWER, UPPER)]))),
            'class_diagnostics': class_diagnostics}


def finish(root):
    started = time.monotonic()
    truths = read(root / 'truth-predictions.json')
    results = []
    for case in read(root / 'protocol.json')['cases']:
        row = {'case_id': case['case_id'], 'truth': case['truth'], 'policies': {}}
        for policy in POLICIES:
            route = root / 'cases' / case['case_id'] / policy
            stages = []
            for r in range(4):
                history = read(route / f'history-{r}.json')
                analysis, key = cached_analysis(root, history)
                if r == 3:
                    write(route / 'fit-3.json', {'cache_key': key, 'cache_path': f'fit-cache/{key}.json'})
                stages.append({'round': r, **stage_metrics(history, analysis, case['truth'], truths[case['case_id']])})
            row['policies'][policy] = {'stages': stages, 'choices': [read(route / f'decision-{r}.json')['candidate_id'] for r in [1, 2, 3]]}
        results.append(row)
    curves = {}
    for policy in POLICIES:
        curves[policy] = []
        for r in range(4):
            records = [row['policies'][policy]['stages'][r] for row in results]
            curves[policy].append({**{key: float(np.mean([s[key] for s in records])) for key in ['heldout_position_rmse_m', 'minimum_sse_per_measurement_m2', 'normalized_parameter_rmse']},
                                   **{key: sum(s[key] for s in records) for key in ['minimum_sse_class_correct', 'class_tie', 'truth_class_competitive']},
                                   'reserved_rmse_m': {name: float(np.mean([s['reserved_rmse_m'][name] for s in records])) for name in physics.RESERVED}})
    write(root / 'results.json', {'cases': results, 'aggregate': {p: curves[p][-1] for p in POLICIES},
                                  'learning_curves': curves, 'interpretation': 'Descriptive eight-case continuous-fit synthetic pilot. Relative fit weights and sensitivity support are heuristic. Two replicates share each class truth; no calibrated uncertainty or significance claim.'})
    physics.trace(root, 'evidence', 'All four equal-measurement policies completed; 24 fresh selector decisions.', [root / 'results.json'])
    write(root / 'stage-finish-runtime.json', {'elapsed_s': time.monotonic() - started})
    write(root / 'manifest.json', {'status': 'complete', 'numerical_runtime_s': runtime(root),
                                    'artifacts': physics.inventory(root, ['manifest.json'])})
    verify(root)
    print('COMPLETE=' + str(root.resolve()), flush=True)


def verify(root):
    protocol = read(root / 'protocol.json')
    for rel, digest in protocol['source_hashes'].items():
        if sha(root / 'sources' / rel) != digest:
            raise ValueError('Frozen source changed: ' + rel)
    for name in ['input-manifest.json', 'prepared-state.json', *[f'round-{r}-{suffix}.json' for r in [1, 2, 3] for suffix in ['frozen', 'complete']], 'manifest.json']:
        if (root / name).exists():
            physics.check_hashes(root, name)
    if (root / 'manifest.json').exists() and set(read(root / 'manifest.json')['artifacts']) != set(physics.inventory(root, ['manifest.json'])):
        raise ValueError('Completed inventory changed')
    for row in map(json.loads, (root / 'trace.jsonl').read_text().splitlines()):
        for rel, digest in row['artifacts'].items():
            if sha(root / rel) != digest:
                raise ValueError('Trace artifact changed: ' + rel)
    runtime(root)
    print('Source/input/stage/artifact hashes verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'acquire', 'verify'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--round', type=int, choices=[1, 2, 3])
    parser.add_argument('--decision-map', type=Path)
    args = parser.parse_args()
    root = (args.out if args.command == 'prepare' else args.campaign).resolve()
    def timeout(signum, frame):
        raise TimeoutError('Numerical command time limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(1200 - (runtime(root) if root.exists() else 0))))
    try:
        if args.command == 'prepare':
            prepare(root)
        elif args.command == 'acquire':
            acquire(root, args.round, args.decision_map)
        else:
            verify(root)
    except TimeoutError:
        raise
    except Exception as exc:
        if root.exists() and not isinstance(exc, (ValueError, FileExistsError)):
            (root / 'failure-stack.txt').write_text(traceback.format_exc())
            write(root / 'failure.json', {'status': 'failed', 'error': str(exc), 'aggregate': 'unavailable_no_subset_ranking'})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
