"""Experimental-design controls for the stick-slip benchmark, paired or on a fresh cohort.

Policies, all using the same frozen fitter and only selector-visible information:
  recipe_astra_modal / recipe_astra_alt  fixed schedules copying the Astra recipe
  eig_mixture        expected information gain over the likelihood-weighted class-mixture support
  t_optimal          model-discrimination (T-type) criterion: class-weight-weighted pairwise squared
                     prediction discrepancy between class best fits, Atkinson-Fedorov sequential form
  t_then_d           discriminate with t_optimal until one class holds >= 0.95 weight, then
                     D-optimal information gain at that class (Box-Hill discriminate-then-estimate)
  fisher_d_classmix / fisher_d_map  D-optimal log-det gain, class-weighted or at the min-SSE class
  numerical_disagreement, fixed_balanced, uniform_random  the sealed campaigns' controls

freeze --out DIR [--fresh --seed S --replicates R]; run --campaign DIR.
On the original 32-system cohort the results are paired against the sealed Astra and Claude
fresh_llm policies and numerical_disagreement is checked byte-for-byte. A fresh cohort draws
32 new systems (8 per class) with R independent noise replicates each, excluding every truth
used before. Support members and class weights are the fitter's heuristics, not posterior samples.
"""
import argparse
import importlib.util
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
REPOSITORY = next(p for p in HERE.parents if (p / 'packages').is_dir() and (p / 'data').is_dir())
CORE_PATH = HERE.with_name('stick_slip_continuous.py')
_spec = importlib.util.spec_from_file_location('design_controls_core', CORE_PATH)
core = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(core)
_espec = importlib.util.spec_from_file_location('design_controls_expanded', HERE.with_name('stick_slip_expanded.py'))
expanded = importlib.util.module_from_spec(_espec); _espec.loader.exec_module(expanded)
np, physics = core.np, core.physics
write, read, sha, now = core.write, core.read, core.sha, core.now
PROFILES, CLASSES = core.PROFILES, core.CLASSES
RECIPES = {'recipe_astra_modal': ['breakaway', 'high_speed', 'release_rest'], 'recipe_astra_alt': ['breakaway', 'release_rest', 'high_speed']}
DESIGN = ['eig_mixture', 't_optimal', 't_then_d', 'fisher_d_classmix', 'fisher_d_map']
CONTROLS = ['numerical_disagreement', 'fixed_balanced', 'uniform_random']
POLICIES = list(RECIPES) + DESIGN + CONTROLS
PAIRED = {'astra': 'stick-slip-expanded-20261008', 'claude': 'stick-slip-claude-expanded-20261009'}
EIG_DRAWS, EIG_SEED, FD_STEP, PRIOR_PRECISION, DOMINANCE = 128, 2026100913, 1e-3, 12.0, 0.95


def mixture(analysis):
    members, weights = [], []
    for label in CLASSES:
        rows = analysis['ensembles'][label]['members']
        if not rows: continue
        sse = np.array([m['standardized_sse'] for m in rows]); w = np.exp(-.5 * (sse - sse.min())); w /= w.sum()
        members.extend(rows); weights.extend(analysis['relative_fit_weights'][label] * w)
    weights = np.array(weights); return members, weights / weights.sum()


def eig_decision(analysis, history):
    members, weights = mixture(analysis)
    rng = np.random.Generator(np.random.PCG64(EIG_SEED + len(history)))
    scores = {}
    for name, profile in PROFILES.items():
        Y = np.array([physics.simulate(m['parameters'], profile)[0] for m in members])
        eps = rng.normal(0., physics.NOISE, (EIG_DRAWS, Y.shape[1])); total = 0.
        for j in range(len(members)):
            d2 = (((Y[j] + eps)[:, None, :] - Y[None, :, :]) ** 2).sum(-1) / physics.NOISE ** 2
            lse = np.logaddexp.reduce(np.log(weights[None, :] + 1e-300) - .5 * d2, axis=1)
            total += weights[j] * float(np.mean(-.5 * d2[:, j] - lse))
        scores[name] = {'eig_nats': total, 'members': len(members)}
    return {'candidate_id': max(PROFILES, key=lambda n: scores[n]['eig_nats']), 'scores': scores,
            'rationale': 'Maximum Monte Carlo expected information gain over the likelihood-weighted class mixture support; listed-order ties.'}


def t_scores(analysis):
    w = analysis['relative_fit_weights']
    pred = {l: {n: physics.simulate(analysis['fits'][l]['best']['parameters'], p)[0] for n, p in PROFILES.items()} for l in CLASSES}
    return {n: float(sum(w[a] * w[b] * np.sum((pred[a][n] - pred[b][n]) ** 2) / physics.NOISE ** 2 for a, b in itertools.combinations(CLASSES, 2))) for n in PROFILES}


def t_decision(analysis):
    s = t_scores(analysis)
    return {'candidate_id': max(PROFILES, key=lambda n: s[n]), 'scores': s,
            'rationale': 'Class-weighted pairwise squared discrepancy between class best-fit predictions (sequential T-type discrimination criterion); listed-order ties.'}


def jacobian(label, z, profile):
    cols = []
    for i in range(4):
        zp, zm = z.copy(), z.copy(); zp[i] = min(1., z[i] + FD_STEP); zm[i] = max(0., z[i] - FD_STEP)
        cols.append((physics.simulate(core.parameters(label, zp), profile)[0] - physics.simulate(core.parameters(label, zm), profile)[0]) / (zp[i] - zm[i]))
    return np.array(cols).T / physics.NOISE


def fisher_gains(analysis, history, labels):
    profiles = core.validate_history(history); out = {}
    for label in labels:
        z = np.array(analysis['fits'][label]['best']['normalized_parameters']); prior = PRIOR_PRECISION * np.eye(4)
        for o in history:
            J = jacobian(label, z, profiles[o['candidate_id']]); prior = prior + J.T @ J
        base = np.linalg.slogdet(prior)[1]
        out[label] = {}
        for name, p in PROFILES.items():
            J = jacobian(label, z, p); out[label][name] = float(np.linalg.slogdet(prior + J.T @ J)[1] - base)
    return out


def fisher_decisions(analysis, history):
    gains = fisher_gains(analysis, history, CLASSES); w = analysis['relative_fit_weights']
    mix = {n: float(sum(w[l] * gains[l][n] for l in CLASSES)) for n in PROFILES}; top = analysis['minimum_sse_class']
    return ({'candidate_id': max(PROFILES, key=lambda n: mix[n]), 'scores': mix, 'per_class': gains, 'rationale': 'Class-weighted D-optimal log-det information gain at class best fits with uniform-prior precision and history information.'},
            {'candidate_id': max(PROFILES, key=lambda n: gains[top][n]), 'scores': gains[top], 'map_class': top, 'rationale': 'D-optimal log-det information gain at the minimum-SSE class best fit only.'})


def t_then_d_decision(analysis, history):
    w = analysis['relative_fit_weights']; top = max(CLASSES, key=lambda l: w[l])
    if w[top] >= DOMINANCE:
        g = fisher_gains(analysis, history, [top])[top]
        return {'candidate_id': max(PROFILES, key=lambda n: g[n]), 'scores': g, 'phase': 'estimate', 'class': top, 'rationale': f'Class {top} holds weight {w[top]:.3f} >= {DOMINANCE}; D-optimal gain at that class.'}
    d = t_decision(analysis); d['phase'] = 'discriminate'; return d


def decide(policy, analysis, history, case, r):
    if policy in RECIPES: return {'candidate_id': RECIPES[policy][r - 1], 'rationale': 'Fixed schedule copying the Astra recipe; no data used.'}
    if policy == 'eig_mixture': return eig_decision(analysis, history)
    if policy == 't_optimal': return t_decision(analysis)
    if policy == 't_then_d': return t_then_d_decision(analysis, history)
    if policy in ('fisher_d_classmix', 'fisher_d_map'): return fisher_decisions(analysis, history)[0 if policy == 'fisher_d_classmix' else 1]
    if policy == 'numerical_disagreement': return core.numerical_decision(analysis)
    if policy == 'fixed_balanced': return {'candidate_id': physics.FIXED[r - 1], 'rationale': 'Frozen balanced sequence'}
    index = int(np.random.default_rng(case['random_seed'] + 100 * r).integers(len(PROFILES)))
    return {'candidate_id': list(PROFILES)[index], 'rationale': 'Seeded uniform random'}


def prior_truths():
    old = [c['truth'] for c in physics.offgrid_cases()] + core.calibration_cases() + [c['truth'] for c in expanded.generated_cases()]
    for seed in [2026100807, 2026100809]:
        z = np.random.Generator(np.random.PCG64(seed)).uniform(.08, .92, (8, 4))
        old += [core.parameters(CLASSES[i // 2], row) for i, row in enumerate(z)]
    return old


def fresh_cases(seed, replicates):
    coordinates = np.random.Generator(np.random.PCG64(seed)).uniform(.08, .92, (32, 4)); old = prior_truths(); cases = []
    for i, z in enumerate(coordinates):
        truth = core.parameters(CLASSES[i // 8], z)
        if truth in old or any(truth == c['truth'] for c in cases): raise RuntimeError('truth collision; no silent resampling')
        for k in range(replicates):
            n = replicates * i + k
            cases.append({'case_id': f'case-{n+1}', 'system_id': f'system-{i+1}', 'replicate': k + 1, 'truth': truth.copy(), 'normalized_truth': z.tolist(),
                          'baseline_seed': 9100000 + n, 'candidate_seed_base': 9200000 + 10000 * n, 'random_seed': 9300000 + n})
    return cases


def freeze(root, fresh=False, seed=None, replicates=1):
    root.mkdir(parents=True, exist_ok=False)
    for name in ['stick_slip_continuous.py', 'stick_slip_offgrid.py', 'stick_slip_expanded.py', 'stick_slip_design_controls.py']:
        dst = root / 'sources/experiments' / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(CORE_PATH.parent / name, dst)
    cases = fresh_cases(seed, replicates) if fresh else expanded.generated_cases()
    paired = {} if fresh else {k: {'campaign': n, 'results_sha256': sha(REPOSITORY / 'data/research' / n / 'results.json')} for k, n in PAIRED.items()}
    write(root / 'protocol.json', {'study': 'Design controls on a fresh cohort' if fresh else 'Design controls paired on the frozen expanded cohort', 'frozen_at': now(),
        'cases': cases, 'fresh': fresh, 'generator': {'seed': seed, 'replicates': replicates, 'bounds_normalized': [.08, .92], 'excluded': 'all prior cohort truths'} if fresh else 'expanded.generated_cases()',
        'fit_config': core.FIT_CONFIG, 'profiles': PROFILES, 'reserved_profiles': physics.RESERVED, 'baseline': physics.BASELINE, 'times': physics.TIMES.tolist(), 'noise': physics.NOISE,
        'policies': POLICIES, 'rounds': 3, 'paired_campaigns': paired,
        'design_settings': {'eig_draws': EIG_DRAWS, 'eig_seed': EIG_SEED, 'fd_step_normalized': FD_STEP, 'prior_precision_normalized': PRIOR_PRECISION, 'dominance_threshold': DOMINANCE},
        'source_hashes': physics.inventory(root / 'sources'),
        'statistics': 'system-level paired differences (replicates averaged within system), class-stratified block bootstrap of 8 systems per class, 10000 draws, seed 2026100812; descriptive only',
        'limitations': 'Support members and class weights are fitter heuristics; EIG, T and Fisher criteria are approximations built on them.'})
    write(root / 'environment.json', {'python': sys.version, 'numpy': np.__version__})


def run(root):
    protocol = read(root / 'protocol.json')
    for name, digest in protocol['source_hashes'].items(): assert sha(root / 'sources' / name) == digest, name
    for key, info in protocol['paired_campaigns'].items(): assert sha(REPOSITORY / 'data/research' / info['campaign'] / 'results.json') == info['results_sha256'], key
    started = time.monotonic(); cases = protocol['cases']
    truths = {c['case_id']: {n: physics.simulate(c['truth'], p)[0].tolist() for n, p in {'baseline': physics.BASELINE, **PROFILES, **physics.RESERVED}.items()} for c in cases}
    write(root / 'truth-predictions.json', truths); results, design_time = [], 0.
    for case in cases:
        observed = (np.array(truths[case['case_id']]['baseline']) + np.random.default_rng(case['baseline_seed']).normal(0, physics.NOISE, 16)).tolist()
        row = {'case_id': case['case_id'], 'system_id': case['system_id'], 'replicate': case.get('replicate', 1), 'truth': case['truth'], 'policies': {}}
        for policy in POLICIES:
            history = [{'candidate_id': 'baseline', 'positions': observed}]; stages, choices = [], []
            for r in range(1, 4):
                analysis, _ = core.cached_analysis(root, history)
                stages.append({'round': r - 1, **core.stage_metrics(history, analysis, case['truth'], truths[case['case_id']])})
                t0 = time.monotonic(); decision = decide(policy, analysis, history, case, r); design_time += time.monotonic() - t0
                choice = decision['candidate_id']; seed = case['candidate_seed_base'] + 100 * r + list(PROFILES).index(choice)
                y = np.array(truths[case['case_id']][choice]) + np.random.default_rng(seed).normal(0, physics.NOISE, 16)
                history = history + [{'candidate_id': choice, 'positions': y.tolist()}]; choices.append(choice)
                write(root / 'cases' / case['case_id'] / policy / f'decision-{r}.json', decision); write(root / 'cases' / case['case_id'] / policy / f'history-{r}.json', history)
            analysis, _ = core.cached_analysis(root, history)
            stages.append({'round': 3, **core.stage_metrics(history, analysis, case['truth'], truths[case['case_id']])})
            row['policies'][policy] = {'stages': stages, 'choices': choices}
            print(f"{case['case_id']} {policy} {choices} rmse={stages[-1]['heldout_position_rmse_m']*1e3:.3f}mm", flush=True)
        results.append(row)
    paired = {}
    for key, info in protocol['paired_campaigns'].items():
        other = read(REPOSITORY / 'data/research' / info['campaign'] / 'results.json'); assert [c['truth'] for c in other['cases']] == [c['truth'] for c in cases]
        paired[key] = {c['case_id']: c['policies'] for c in other['cases']}
    for row in results:
        for key in paired: row['policies'][f'{key}_fresh_llm'] = paired[key][row['case_id']]['fresh_llm']
        if 'astra' in paired:
            row['numerical_disagreement_matches_sealed'] = abs(paired['astra'][row['case_id']]['numerical_disagreement']['stages'][-1]['heldout_position_rmse_m'] - row['policies']['numerical_disagreement']['stages'][-1]['heldout_position_rmse_m']) < 1e-12
    all_policies = POLICIES + [f'{k}_fresh_llm' for k in paired]
    final = lambda row, p: row['policies'][p]['stages'][-1]['heldout_position_rmse_m']
    systems = list(dict.fromkeys(r['system_id'] for r in results))
    sysmean = {p: np.array([np.mean([final(r, p) for r in results if r['system_id'] == s]) for s in systems]) for p in all_policies}
    sysclass = [next(r['truth']['class'] for r in results if r['system_id'] == s) for s in systems]
    aggregate = {p: {'heldout_position_rmse_m': float(sysmean[p].mean()), 'normalized_parameter_rmse': float(np.mean([r['policies'][p]['stages'][-1]['normalized_parameter_rmse'] for r in results])),
                     'minimum_sse_class_correct': int(sum(r['policies'][p]['stages'][-1]['minimum_sse_class_correct'] for r in results)), 'cases': len(results)} for p in all_policies}
    rng = np.random.Generator(np.random.PCG64(2026100812))
    blocks = [np.array([i for i, c in enumerate(sysclass) if c == l]) for l in CLASSES]
    idx = np.concatenate([b[rng.integers(0, len(b), size=(10000, len(b)))] for b in blocks], axis=1)
    comparisons = {}
    for p in all_policies:
        for q in all_policies:
            if p == q: continue
            d = sysmean[p] - sysmean[q]
            comparisons[f'{p}_minus_{q}'] = {'mean_difference_m': float(d.mean()), 'descriptive_block_bootstrap_95pct_m': np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist(), 'systems_lower': int((d < -1e-12).sum()), 'systems_higher': int((d > 1e-12).sum()), 'independent_units': len(systems)}
    write(root / 'results.json', {'cases': results, 'aggregate': aggregate, 'paired_comparisons': comparisons, 'systems': len(systems), 'replicates_per_system': len(results) // len(systems),
                                  'numerical_disagreement_replicated_byte_level': all(r.get('numerical_disagreement_matches_sealed', True) for r in results) if paired else None,
                                  'design_computation_s': design_time, 'total_runtime_s': time.monotonic() - started,
                                  'interpretation': 'Descriptive system-level comparison of fixed recipes and approximate design criteria built on the frozen fitter support; no calibrated significance claim.'})
    write(root / 'manifest.json', {'status': 'complete', 'artifacts': physics.inventory(root, ['manifest.json'])}); print('COMPLETE=' + str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run']); parser.add_argument('--out', type=Path); parser.add_argument('--campaign', type=Path)
    parser.add_argument('--fresh', action='store_true'); parser.add_argument('--seed', type=int); parser.add_argument('--replicates', type=int, default=1)
    a = parser.parse_args()
    if a.command == 'freeze':
        if a.fresh and a.seed is None: parser.error('--fresh requires --seed')
        freeze(a.out.resolve(), a.fresh, a.seed, a.replicates)
    else: run(a.campaign.resolve())


if __name__ == '__main__':
    main()
