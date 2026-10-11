"""Class-agnostic complementarity time-stepping estimator on recipe-collected histories.

For each case: load the 64-point history collected by a chosen policy in a completed design
campaign, fit (friction, restitution, spring, damping, threshold ratio, restitution slope) by
minimising standardised position misfit through the differentiable stepper with a batched
multistart (Adam then L-BFGS), and evaluate parameter error, implied class (ratio > 1.3 ->
static, slope > 0.05 -> speed), and reserved-profile error by rolling the learned model out
with its own stepper against the analytic truth. Classical fitter numbers are copied alongside.

Run: .venv/bin/python experiments/slider_complementarity_pilot.py --campaign DIR --policy recipe_astra_modal --cases case-1 ... --out data/research/NEW
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from packages.dynamics import slider_timestepping as ts


def load_physics(campaign):
    spec = importlib.util.spec_from_file_location('pilot_physics', campaign / 'sources/experiments/stick_slip_offgrid.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def fit(history, profiles_by_name, times, starts=4, adam_steps=300, lbfgs_steps=100, seed=0, h=2e-3):
    torch.manual_seed(seed); torch.set_num_threads(2)
    profiles = torch.as_tensor([profiles_by_name[o['candidate_id']] for o in history], dtype=ts.DTYPE)
    y = torch.as_tensor([o['positions'] for o in history], dtype=ts.DTYPE)
    raw = torch.nn.Parameter(torch.randn(starts, 6, dtype=ts.DTYPE) * 1.0)
    tt = torch.as_tensor(times, dtype=ts.DTYPE)
    def losses():
        pred = ts.simulate(ts.unpack(raw), profiles, tt, h=h)
        return ((pred - y[None]) ** 2).sum(dim=(1, 2)) / ts.NOISE ** 2 if hasattr(ts, 'NOISE') else ((pred - y[None]) ** 2).sum(dim=(1, 2)) / 0.003 ** 2
    opt = torch.optim.Adam([raw], lr=0.05); trace = []
    for step in range(adam_steps):
        opt.zero_grad(); L = losses(); L.sum().backward(); opt.step()
        if step % 50 == 0: trace.append({'step': step, 'sse': L.detach().tolist()})
    lb = torch.optim.LBFGS([raw], max_iter=lbfgs_steps, line_search_fn='strong_wolfe', tolerance_grad=1e-10, tolerance_change=1e-12)
    def closure():
        lb.zero_grad(); L = losses().sum(); L.backward(); return L
    lb.step(closure)
    L = losses().detach(); best = int(torch.argmin(L))
    p = {k: float(v[best]) for k, v in ts.unpack(raw.detach()).items()}
    return p, {'standardized_sse': float(L[best]), 'all_starts_sse': L.tolist(), 'trace': trace}, raw.detach()[best]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True); ap.add_argument('--policy', default='recipe_astra_modal'); ap.add_argument('--cases', nargs='+', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--starts', type=int, default=4); ap.add_argument('--adam-steps', type=int, default=300); ap.add_argument('--lbfgs-steps', type=int, default=100); ap.add_argument('--h', type=float, default=2e-3)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    physics = load_physics(a.campaign); results = json.load(open(a.campaign / 'results.json')); rows = {r['case_id']: r for r in results['cases']}
    profiles = {'baseline': physics.BASELINE, **physics.PROFILES}; times = physics.TIMES.tolist(); out = []
    for case_id in a.cases:
        row = rows[case_id]; truth = row['truth']; classical = row['policies'][a.policy]['stages'][-1]
        history = json.load(open(a.campaign / 'cases' / case_id / a.policy / 'history-3.json'))
        t0 = time.time(); p, diag, raw = fit(history, profiles, times, a.starts, a.adam_steps, a.lbfgs_steps, h=a.h); elapsed = time.time() - t0
        implied = ('static' if p['ratio'] > 1.3 else 'coulomb') + '_' + ('speed' if p['slope'] > 0.05 else 'constant')
        # reserved-profile error: learned model rolled out with its own stepper vs analytic truth
        with torch.no_grad():
            params = {k: torch.tensor([v], dtype=ts.DTYPE) for k, v in p.items()}
            res_profiles = torch.as_tensor([physics.RESERVED[n] for n in physics.RESERVED], dtype=ts.DTYPE)
            pred = ts.simulate(params, res_profiles, torch.as_tensor(times, dtype=ts.DTYPE), h=a.h)[0].numpy()
        reserved = {n: float(np.sqrt(np.mean((pred[i] - physics.simulate(truth, physics.RESERVED[n])[0]) ** 2))) for i, n in enumerate(physics.RESERVED)}
        # same learned continuous parameters evaluated with the analytic simulator under the implied class (comparable to the classical metric)
        learned_cls = {'class': implied, 'friction': p['friction'], 'restitution': p['restitution'], 'spring': p['spring'], 'damping': p['damping']}
        reserved_analytic = {n: float(np.sqrt(np.mean((physics.simulate(learned_cls, pr)[0] - physics.simulate(truth, pr)[0]) ** 2))) for n, pr in physics.RESERVED.items()}
        perr = {k: p[k] - truth[k] for k in ['friction', 'restitution', 'spring', 'damping']}
        truth_ratio, truth_slope = (1.6 if truth['class'].startswith('static') else 1.0), (0.10 if truth['class'].endswith('speed') else 0.0)
        result = {'case_id': case_id, 'truth': truth, 'truth_ratio': truth_ratio, 'truth_slope': truth_slope,
                  'complementarity': {'parameters': p, 'implied_class': implied, 'class_correct': implied == truth['class'], 'parameter_error': perr,
                                      'normalized_parameter_rmse': float(np.sqrt(np.mean([(perr[k] / (hi - lo)) ** 2 for k, (lo, hi) in list(ts.BOUNDS.items())[:4]]))),
                                      'reserved_rmse_m_stepper': reserved, 'heldout_position_rmse_m_stepper': float(np.sqrt(np.mean([v ** 2 for v in reserved.values()]))),
                                      'reserved_rmse_m_analytic': reserved_analytic, 'heldout_position_rmse_m': float(np.sqrt(np.mean([v ** 2 for v in reserved_analytic.values()]))),
                                      'diagnostics': diag, 'elapsed_s': elapsed, 'settings': {'starts': a.starts, 'adam_steps': a.adam_steps, 'lbfgs_steps': a.lbfgs_steps, 'h': a.h}},
                  'classical': {'selected_class': classical['minimum_sse_class'], 'class_correct': classical['minimum_sse_class_correct'], 'normalized_parameter_rmse': classical['normalized_parameter_rmse'],
                                'reserved_rmse_m': classical['reserved_rmse_m'], 'heldout_position_rmse_m': classical['heldout_position_rmse_m'], 'parameters': classical['class_diagnostics'][classical['minimum_sse_class']]['parameters']}}
        json.dump(result, open(a.out / f'{case_id}.json', 'w'), indent=2); out.append(result)
        c = result['complementarity']
        print(f"{case_id} {truth['class']} -> {implied} ok={c['class_correct']} ratio={p['ratio']:.2f} slope={p['slope']:.3f} | heldout analytic={c['heldout_position_rmse_m']*1e3:.3f}mm stepper={c['heldout_position_rmse_m_stepper']*1e3:.3f}mm classical={classical['heldout_position_rmse_m']*1e3:.3f}mm | param rmse {c['normalized_parameter_rmse']:.3f} vs {classical['normalized_parameter_rmse']:.3f} | sse {diag['standardized_sse']:.1f} ({elapsed:.0f}s)", flush=True)
    summary = {'cases': len(out), 'class_correct': {'complementarity': sum(r['complementarity']['class_correct'] for r in out), 'classical': sum(r['classical']['class_correct'] for r in out)},
               'mean_heldout_rmse_m': {'complementarity_analytic': float(np.mean([r['complementarity']['heldout_position_rmse_m'] for r in out])), 'complementarity_stepper': float(np.mean([r['complementarity']['heldout_position_rmse_m_stepper'] for r in out])), 'classical': float(np.mean([r['classical']['heldout_position_rmse_m'] for r in out]))},
               'mean_normalized_parameter_rmse': {'complementarity': float(np.mean([r['complementarity']['normalized_parameter_rmse'] for r in out])), 'classical': float(np.mean([r['classical']['normalized_parameter_rmse'] for r in out]))},
               'ratio_slope_recovery': [{'case': r['case_id'], 'truth': [r['truth_ratio'], r['truth_slope']], 'learned': [round(r['complementarity']['parameters']['ratio'], 3), round(r['complementarity']['parameters']['slope'], 3)]} for r in out]}
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=2); print(json.dumps({k: v for k, v in summary.items() if k != 'ratio_slope_recovery'}, indent=1))


if __name__ == '__main__':
    main()
