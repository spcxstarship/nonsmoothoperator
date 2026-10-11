"""Prospective revalidation after an event-solver and finite-candidate retention repair.

freeze creates the immutable protocol before any new-system performance is seen.
preflight checks only physics. prepare/acquire are explicitly separate fit stages.
Run subsequent commands with the frozen sources/experiments/stick_slip_revalidation.py.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
REPOSITORY = HERE.parents[1]
CORE_PATH = HERE.with_name('stick_slip_continuous.py')
DEPENDENCIES = ['stick_slip_continuous.py', 'stick_slip_offgrid.py']
_spec = importlib.util.spec_from_file_location('validation_frozen_continuous', CORE_PATH)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
np, physics = core.np, core.physics
write, read, sha, now = core.write, core.read, core.sha, core.now
POLICIES, PROFILES = core.POLICIES, core.PROFILES
cached_analysis, stage_metrics, runtime = core.cached_analysis, core.stage_metrics, core.runtime


def generated_cases():
    rng = np.random.Generator(np.random.PCG64(2026100809))
    coordinates = rng.uniform(.08, .92, (8, 4))
    old = [c['truth'] for c in physics.offgrid_cases()] + core.calibration_cases()
    previous = np.random.Generator(np.random.PCG64(2026100807)).uniform(.08, .92, (8, 4))
    old += [core.parameters(core.CLASSES[i // 2], z) for i, z in enumerate(previous)]
    cases = []
    for i, z in enumerate(coordinates):
        truth = core.parameters(core.CLASSES[i // 2], z)
        if truth in old:
            raise RuntimeError('Exact prior truth collision; no silent rejection/resampling')
        for replicate in range(1, 4):
            index = 3 * i + replicate - 1
            cases.append({'case_id': f'case-{index+1}', 'system_id': f'system-{i+1}',
                          'replicate': replicate, 'truth': truth.copy(),
                          'normalized_truth': z.tolist(), 'baseline_seed': 7100000+index,
                          'candidate_seed_base': 7200000+10000*index,
                          'random_seed': 7300000+index})
    return cases


def freeze(root, analysis_dir=None):
    root.mkdir(parents=True, exist_ok=False)
    copies = {f'experiments/{name}': CORE_PATH.parent / name for name in DEPENDENCIES}
    copies.update({'experiments/stick_slip_revalidation.py': HERE,
                   'tests/test_stick_slip_revalidation.py': REPOSITORY / 'tests/test_stick_slip_revalidation.py',
                   'tests/test_stick_slip_continuous.py': REPOSITORY / 'tests/test_stick_slip_continuous.py',
                   'tests/test_stick_slip_event_regression.py': REPOSITORY / 'tests/test_stick_slip_event_regression.py'})
    if analysis_dir is not None:
        for name in ['record_selectors.py', 'verify_independent.py', 'check_fitted_reference.py', 'report_results.py']:
            copies['analysis/' + name] = analysis_dir / name
    for relative, source in copies.items():
        destination = root / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    protocol = {
        'study': 'Prospective new-parameter validation after event/refit repair', 'frozen_at': now(),
        'cases': generated_cases(), 'fit_config': core.FIT_CONFIG,
        'profiles': PROFILES, 'reserved_profiles': physics.RESERVED,
        'baseline': physics.BASELINE, 'times': physics.TIMES.tolist(), 'noise': physics.NOISE,
        'policies': POLICIES, 'fixed_sequence': physics.FIXED, 'rounds': 3,
        'measurement_budget': 64, 'maximum_selector_calls': 72, 'unique_systems': 8,
        'noise_replicates_per_system': 3, 'numerical_runtime_s': 1200,
        'selector_instructions': core.INSTRUCTIONS,
        'analysis_sources': [name for name in copies if name.startswith('analysis/')],
        'source_hashes': physics.inventory(root / 'sources'),
        'generator': {'bit_generator': 'PCG64', 'seed': 2026100809,
                      'normalized_uniform_bounds': [.08, .92], 'shape': [8, 4],
                      'assignment': 'class-major; two independent parameter vectors per class; three noise replicates per vector',
                      'selection': 'No performance selection, rejection, or tuning'},
        'primary_metric': 'Mean case-level aggregate reserved-position RMSE in meters',
        'statistics': {'pairing': 'same case and shared candidate/round noise across policies',
                       'system_summary': 'average three noise replicates for each of eight systems',
                       'primary_contrast': 'fresh_llm minus fixed_balanced mean reserved RMSE; numerical_disagreement and uniform_random secondary',
                       'bootstrap': 'paired system-block resampling stratified by class: resample two system means with replacement within each of four classes; 10000 draws; descriptive percentile 95% interval',
                       'bootstrap_seed': 2026100810},
        'development_disclosure': 'Event/refit robustness developed using calibration cases and the failed 20261008 validation cohort. All those systems are excluded from this new cohort. New orchestration SMOKE uses calibration only. No new-cohort fits or selector outcomes examined before freeze.',
        'limitations': 'Eight synthetic systems, three noise repetitions each; frozen mechanics and local fitter. No novel mechanism discovery or calibrated significance claim.',
        'secondary_finite_bank_replay': 'Reuse old 32-particle Bayesian inference on exact new fixed_balanced and uniform_random observed histories and same three reserved profiles to assess continuous fitting gains; no new acquisition, training, or heldout-data fitting. Analysis code prepared before performance evaluation and executed only after sealing the complete campaign.',
        'failure_policy': 'Report unresolved fits without retuning; retain directly verified finite candidates when local refinement fails. Stop if no finite fit exists for any class, finite-candidate reevaluation fails, or a campaign stage fails. Failed execution does not yield a subset ranking.'}
    write(root / 'protocol.json', protocol)
    write(root / 'generator-manifest.json', {'generator': protocol['generator'], 'cases': protocol['cases'],
                                           'source_hashes': protocol['source_hashes']})
    write(root / 'environment.json', {'python': sys.version, 'numpy': np.__version__, 'device': 'cpu'})
    write(root / 'input-manifest.json', {'artifacts': physics.inventory(root)})
    physics.trace(root, 'freeze', 'Generator, complete source dependencies, cases, seeds and protocol frozen before performance evaluation.', [root / 'protocol.json', root / 'generator-manifest.json'])
    verify(root)


def preflight(root):
    verify(root)
    start = time.monotonic()
    refinement, references = [], []
    for case in read(root / 'protocol.json')['cases'][::3]:
        for name, profile in {'baseline': physics.BASELINE, **PROFILES, **physics.RESERVED}.items():
            p = case['truth']
            y, events = physics.simulate(p, profile)
            refined, refined_events = physics.simulate(p, profile, collapse_speed=2.5e-7, root_tolerance=2.5e-13)
            reference, impacts = physics.reference(p, profile, physics.TIMES)
            actual = [(e['time'], e['kind']) for e in events if e['kind'] in ['left', 'right']]
            refinement.append({'system_id': case['system_id'], 'profile': name,
                               'max_position_change': float(np.max(abs(y-refined))),
                               'event_kind_sequence_same': [e['kind'] for e in events] == [e['kind'] for e in refined_events]})
            references.append({'system_id': case['system_id'], 'profile': name,
                               'position_max_error_m': float(np.max(abs(y-reference))),
                               'impact_sequence_identical': [e[1] for e in actual] == [e[1] for e in impacts],
                               'impact_time_max_error_s': max((abs(a[0]-b[0]) for a,b in zip(actual, impacts)), default=0.)})
    write(root / 'contact-refinement.json', refinement)
    write(root / 'reference-verification.json', references)
    passed = (all(r['max_position_change'] <= 1e-5 and r['event_kind_sequence_same'] for r in refinement)
              and all(r['position_max_error_m'] <= 1e-7 and r['impact_time_max_error_s'] <= 1e-7 and r['impact_sequence_identical'] for r in references))
    write(root / 'preflight-state.json', {'passed': passed, 'artifacts': {name: sha(root / name) for name in ['contact-refinement.json', 'reference-verification.json']}})
    write(root / 'stage-preflight-runtime.json', {'elapsed_s': time.monotonic()-start})
    if not passed:
        raise RuntimeError('Physics reference/refinement gate failed; no validation fits run')


def prepare(root):
    verify(root)
    if not read(root / 'preflight-state.json')['passed']:
        raise ValueError('Physics preflight must pass before prepare')
    start = time.monotonic()
    cases = read(root / 'protocol.json')['cases']
    truths = {case['case_id']: {name: physics.simulate(case['truth'], profile)[0].tolist()
              for name, profile in {'baseline': physics.BASELINE, **PROFILES, **physics.RESERVED}.items()} for case in cases}
    write(root / 'truth-predictions.json', truths)
    for case in cases:
        observed = np.array(truths[case['case_id']]['baseline']) + np.random.default_rng(case['baseline_seed']).normal(0, physics.NOISE, 16)
        for policy in POLICIES:
            write(root / 'cases' / case['case_id'] / policy / 'history-0.json', [{'candidate_id': 'baseline', 'positions': observed.tolist()}])
    write(root / 'prepared-state.json', {'artifacts': {str(p.relative_to(root)): sha(p) for p in root.rglob('*.json') if p.name in ['truth-predictions.json', 'history-0.json']}})
    write(root / 'stage-prepare-runtime.json', {'elapsed_s': time.monotonic()-start})
    freeze_round(root, 1)


def freeze_round(root, r):
    verify(root)
    core.freeze_round(root, r)


def acquire(root, r, map_path):
    if (root / f'round-{r}-lock.json').exists():
        raise ValueError('Round already started; no retries')
    if r > 1 and not (root / f'round-{r-1}-complete.json').exists():
        raise ValueError('Previous round incomplete')
    verify(root)
    entries = read(map_path)
    expected = {row['case_id'] for row in read(root / f'round-{r}-contexts.json')}
    if expected != {case['case_id'] for case in read(root / 'protocol.json')['cases']}:
        raise ValueError('Context ledger must contain every protocol case')
    if len(entries) != len(expected) or {row['case_id'] for row in entries} != expected:
        raise ValueError('Require one unique actual fresh selector decision per validation case')
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


def finish(root):
    started = time.monotonic()
    truths = read(root / 'truth-predictions.json')
    results = []
    for case in read(root / 'protocol.json')['cases']:
        row = {'case_id': case['case_id'], 'truth': case['truth'], 'system_id': case['system_id'], 'replicate': case['replicate'], 'policies': {}}
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
    system_summaries = []
    for system_id in dict.fromkeys(row['system_id'] for row in results):
        rows = [row for row in results if row['system_id'] == system_id]
        system_summaries.append({'system_id': system_id, 'class': rows[0]['truth']['class'], 'policies': {p: float(np.mean([row['policies'][p]['stages'][-1]['heldout_position_rmse_m'] for row in rows])) for p in POLICIES}})
    comparisons = {}
    rng = np.random.Generator(np.random.PCG64(2026100810))
    blocks = [np.array([i for i, row in enumerate(system_summaries) if row['class'] == label]) for label in core.CLASSES]
    indices = np.concatenate([block[rng.integers(0, len(block), size=(10000, len(block)))] for block in blocks if len(block)], axis=1)
    for control in POLICIES[1:]:
        delta = np.array([row['policies']['fresh_llm']-row['policies'][control] for row in system_summaries])
        case_delta = [row['policies']['fresh_llm']['stages'][-1]['heldout_position_rmse_m']-row['policies'][control]['stages'][-1]['heldout_position_rmse_m'] for row in results]
        comparisons[control] = {'sign': 'fresh_llm minus control; negative favors fresh_llm', 'paired_case_differences_m': case_delta, 'paired_system_differences_m': delta.tolist(), 'mean_difference_m': float(delta.mean()), 'descriptive_block_bootstrap_95pct_m': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(), 'independent_units': len(system_summaries)}
    write(root / 'results.json', {'cases': results, 'aggregate': {p: curves[p][-1] for p in POLICIES},
                                  'learning_curves': curves, 'system_summaries': system_summaries, 'paired_comparisons': comparisons, 'interpretation': 'Descriptive 24-case, eight-system continuous-fit synthetic validation. Relative fit weights and sensitivity support are heuristic. Three noise replicates share each system truth; no calibrated uncertainty or significance claim.'})
    physics.trace(root, 'evidence', 'All four equal-measurement policies completed; 72 fresh selector decisions.', [root / 'results.json'])
    write(root / 'stage-finish-runtime.json', {'elapsed_s': time.monotonic() - started})
    write(root / 'manifest.json', {'status': 'complete', 'numerical_runtime_s': runtime(root),
                                    'artifacts': physics.inventory(root, ['manifest.json'])})
    verify(root)
    print('COMPLETE=' + str(root.resolve()), flush=True)


def verify(root):
    core.verify(root)
    if (root / 'preflight-state.json').exists():
        physics.check_hashes(root, 'preflight-state.json')
    protocol = read(root / 'protocol.json')
    # Ensure the code actually executing is the code sealed by the protocol.
    for source in [HERE, CORE_PATH, CORE_PATH.with_name('stick_slip_offgrid.py')]:
        key = 'experiments/' + source.name
        if sha(source) != protocol['source_hashes'][key]:
            raise ValueError('Executing source differs from frozen source: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'preflight', 'prepare', 'acquire', 'verify'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--round', type=int, choices=[1, 2, 3])
    parser.add_argument('--decision-map', type=Path)
    parser.add_argument('--analysis-dir', type=Path)
    args = parser.parse_args()
    root = (args.out if args.command == 'freeze' else args.campaign).resolve()
    def timeout(signum, frame):
        raise TimeoutError('1200-second active numerical budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(1200-(runtime(root) if root.exists() else 0))))
    started = time.monotonic()
    try:
        if args.command == 'acquire':
            acquire(root, args.round, args.decision_map)
        elif args.command == 'freeze':
            if args.analysis_dir is None:
                parser.error('freeze requires --analysis-dir to seal selector recording and analysis code')
            freeze(root, args.analysis_dir)
        else:
            globals()[args.command](root)
    except Exception as exc:
        if root.exists() and not isinstance(exc, (ValueError, FileExistsError)):
            (root / 'failure-stack.txt').write_text(traceback.format_exc())
            if not (root / 'failure.json').exists():
                write(root / 'failure.json', {'status': 'failed', 'error': str(exc), 'aggregate': 'unavailable_no_subset_ranking', 'failed_command_elapsed_s': time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
