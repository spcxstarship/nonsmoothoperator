"""Slider PINN pilot: fit the regularised-inclusion PINN on recipe-collected histories.

Reads the final 64-point histories collected by a chosen policy in a completed
design-controls campaign, trains packages/pinn/slider.py with the class selected by the
classical fitter (selector-visible information only), and evaluates: learned parameter
error, dense reconstruction error on the training runs, and reserved-profile prediction
error by simulating the learned parameters in the event-driven simulator. The classical
fitter's numbers for the same histories are copied alongside for comparison.

Run: .venv/bin/python experiments/slider_pinn_pilot.py --campaign DIR --policy recipe_astra_modal
     --cases case-1 case-2 ... --out data/research/NEW
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from packages.pinn import slider


def load_physics(campaign):
    spec = importlib.util.spec_from_file_location('pilot_physics', campaign / 'sources/experiments/stick_slip_offgrid.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True); p.add_argument('--policy', default='recipe_astra_modal')
    p.add_argument('--cases', nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--adam-steps', type=int, default=2000); p.add_argument('--lbfgs-steps', type=int, default=300); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--residual-weight', type=float, default=1.0); p.add_argument('--wall-free-residual', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    physics = load_physics(a.campaign)
    results = json.load(open(a.campaign / 'results.json'))
    rows = {r['case_id']: r for r in results['cases']}
    profiles = {'baseline': physics.BASELINE, **physics.PROFILES}
    out = []
    for case_id in a.cases:
        row = rows[case_id]; truth = row['truth']
        history = json.load(open(a.campaign / 'cases' / case_id / a.policy / 'history-3.json'))
        classical = row['policies'][a.policy]['stages'][-1]
        label = classical['minimum_sse_class']          # selector-visible: the classical fitter's selected class
        fit = slider.train(history, profiles, label, seed=a.seed, adam_steps=a.adam_steps, lbfgs_steps=a.lbfgs_steps, residual_weight=a.residual_weight, wall_free_residual=a.wall_free_residual)
        learned = {'class': label, **fit['parameters']}
        reserved = {}
        for name, profile in physics.RESERVED.items():
            pred = physics.simulate(learned, profile)[0]; true = physics.simulate(truth, profile)[0]
            reserved[name] = float(np.sqrt(np.mean((pred - true) ** 2)))
        dense = np.array(fit['dense_times'])
        recon = []
        for i, run in enumerate(history):
            true_dense = physics.simulate(truth, profiles[run['candidate_id']], times=dense)[0]
            recon.append({'candidate_id': run['candidate_id'], 'rmse_m': float(np.sqrt(np.mean((np.array(fit['reconstruction'][i]) - true_dense) ** 2)))})
        perr = {k: learned[k] - truth[k] for k in slider.BOUNDS}
        result = {'case_id': case_id, 'truth': truth, 'selected_class': label, 'class_correct': label == truth['class'],
                  'pinn': {'parameters': fit['parameters'], 'parameter_error': perr,
                           'normalized_parameter_rmse': float(np.sqrt(np.mean([(perr[k] / (hi - lo)) ** 2 for k, (lo, hi) in slider.BOUNDS.items()]))),
                           'reserved_rmse_m': reserved, 'heldout_position_rmse_m': float(np.sqrt(np.mean([v ** 2 for v in reserved.values()]))),
                           'reconstruction_rmse_m': recon, 'final_loss': fit['final_loss'], 'elapsed_s': fit['elapsed_s'], 'settings': fit['settings'], 'trace': fit['trace']},
                  'classical': {'parameters': classical['class_diagnostics'][label]['parameters'], 'normalized_parameter_rmse': classical['normalized_parameter_rmse'],
                                'reserved_rmse_m': classical['reserved_rmse_m'], 'heldout_position_rmse_m': classical['heldout_position_rmse_m']}}
        json.dump(result, open(a.out / f'{case_id}.json', 'w'), indent=2)
        out.append(result)
        print(f"{case_id} {truth['class']} class_ok={label == truth['class']} pinn={result['pinn']['heldout_position_rmse_m']*1e3:.3f}mm classical={classical['heldout_position_rmse_m']*1e3:.3f}mm "
              f"param_rmse pinn={result['pinn']['normalized_parameter_rmse']:.3f} classical={classical['normalized_parameter_rmse']:.3f} ({fit['elapsed_s']:.0f}s)", flush=True)
    summary = {'campaign': str(a.campaign), 'policy': a.policy, 'cases': len(out),
               'mean_heldout_rmse_m': {'pinn': float(np.mean([r['pinn']['heldout_position_rmse_m'] for r in out])), 'classical': float(np.mean([r['classical']['heldout_position_rmse_m'] for r in out]))},
               'mean_normalized_parameter_rmse': {'pinn': float(np.mean([r['pinn']['normalized_parameter_rmse'] for r in out])), 'classical': float(np.mean([r['classical']['normalized_parameter_rmse'] for r in out]))},
               'per_parameter_mean_abs_error': {k: {'pinn': float(np.mean([abs(r['pinn']['parameter_error'][k]) for r in out])), 'classical': float(np.mean([abs(r['classical']['parameters'][k] - r['truth'][k]) for r in out]))} for k in slider.BOUNDS},
               'variant': {'residual_weight': a.residual_weight, 'wall_free_residual': a.wall_free_residual}, 'interpretation': 'Pilot on recipe-collected histories; class taken from the classical fitter; one seed; smoothing constants declared, not tuned.'}
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=2); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
