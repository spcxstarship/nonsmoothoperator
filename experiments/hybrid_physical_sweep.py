"""Bounded physical-setting sweep: adaptive information vs random vs fixed downward.

Run: .venv/bin/python experiments/hybrid_physical_sweep.py run --out data/research/NEW
Verify using NEW/sources/experiments/hybrid_physical_sweep.py verify --campaign NEW.
Twelve trials, four independent training jobs each, 50s/trial, no budget borrowing.
No LLM selection is performed or represented in this campaign.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np
import scipy
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import hybrid_selection_round as core

SELF='experiments/hybrid_physical_sweep.py'
POLICIES=['adaptive_numerical','uniform_random','fixed_downward']
PRIMARY='parameter_transfer_position_rmse_m'


def design():
    rows=[]
    for floor in [.03,.12,.22]:
        for restitution in [.65,.90]:
            for replicate in [1,2]:
                index=len(rows)
                rows.append({'trial_id':f'trial-{index+1:02d}','setting_id':f'floor-{floor:.2f}-e-{restitution:.2f}',
                             'floor_m':floor,'restitution':restitution,'noise_replicate':replicate,
                             'baseline_noise_seed':100001+index,'random_policy_seed':300000+index,
                             'candidate_noise_seeds':{identifier:200000+1000*index+offset for identifier,_,offset in core.CANDIDATES}})
    return rows


def common():
    return {'simulator':{'floor_m':.12,'restitution':.8,'gravity_m_s2':9.81},
            'inference':{'floor_bounds_m':[0.,.25],'restitution_bounds':[.6,.95],'initial_floor_m':.05,'initial_restitution':.65},
            'pinn':{'width':16,'collocation_points_per_segment':48,'adam_steps':1500,'adam_learning_rate':.003,
                    'lbfgs_steps':1000,'physics_weight':100.,'guard_weight':100.},
            'data':{'initial_height_m':1.,'times_s':core.TIMES,'noise_std_m':.01,
                    'candidate_seed_offsets':{identifier:offset for identifier,_,offset in core.CANDIDATES}},
            'evaluation':{'initial_heights_m':[.85,1.15],'accelerations_m_s2':[-2.,2.],'horizon_s':.62,
                          'points_per_trajectory':301,'primary':PRIMARY,'definition':'Analytic parameter transfer, NOT direct neural generalization'},
            'budgets':{'maximum_training_jobs':4,'job_timeout_s':120,'total_compute_s':50,'cleanup_reserve_s':.5},
            'information_selector':{'coordinates':'parameters normalized by prior ranges','difference_step':1e-5,
                                    'noise_std':.01,'jitter':1e-9,'tie_break':'listed candidate order'},'model_seed':0}


def validate_horizons(config):
    prior=config['inference']
    # Entire inference prior supports a single impact for every candidate.
    for _,acceleration,_ in core.CANDIDATES:
        obs=core.Observations(1.,acceleration,np.array(core.TIMES),np.zeros(12))
        core.validate_observations([obs],prior['floor_bounds_m'],prior['restitution_bounds'],9.81)
    for h0 in [.85,1.15]:
        for acceleration in [-2.,2.]:
            obs=core.Observations(h0,acceleration,np.linspace(0,.62,301),np.zeros(301))
            core.validate_observations([obs],prior['floor_bounds_m'],prior['restitution_bounds'],9.81)
    for spec in design():
        for _,acceleration,_ in core.CANDIDATES:
            core.predict(spec['floor_m'],spec['restitution'],1.,acceleration,np.array(core.TIMES))


def source_check(root):
    for name,digest in core.read(root/'protocol.json')['source_sha256'].items():
        if core.sha(root/'sources'/name)!=digest:raise ValueError(f'frozen source mismatch: {name}')


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    config=common();validate_horizons(config)
    names=core.SOURCE_NAMES+[SELF]
    sources=root/'sources'
    for name in names:
        target=sources/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    hashes={name:core.sha(sources/name) for name in names}
    protocol={'study':'six physical settings with two independent noise replicates each','frozen_at_utc':core.now(),
              'common':config,'trial_specs':design(),'policies':POLICIES,'source_sha256':hashes,
              'maximum_training_jobs':48,'maximum_total_compute_s':600,'budget_borrowing':False,
              'random_rule':'Independent numpy.default_rng(random_policy_seed).integers(0,3), listed candidate order; no balancing.',
              'failure_rule':'Retain all trials; any incomplete fit suppresses all aggregate rankings and paired summaries.',
              'analysis':'Per-setting means of two noise replicates, all12 paired policy differences and six setting-cluster means; descriptive only, no hypothesis tests.',
              'policy_identity':'No LLM choices: adaptive_numerical, uniform_random, fixed_downward.'}
    core.write(root/'protocol.json',protocol)
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
         'device':'cpu','dtype':'float64','threads':1,'invocation':sys.argv}
    core.write(root/'environment.json',env)
    trials=root/'trials';trials.mkdir()
    # Freeze every child's source, truth, seed, baseline dataset and budget before any job.
    for spec in protocol['trial_specs']:
        trial=trials/spec['trial_id'];trial.mkdir()
        for name in names:
            target=trial/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(sources/name,target)
        child=common();child['simulator'].update({'floor_m':spec['floor_m'],'restitution':spec['restitution']})
        child.update({'study':'pure numerical physical adaptation trial','trial_spec':spec,'source_sha256':hashes,'frozen_at_utc':protocol['frozen_at_utc']})
        child['data'].update({'baseline_seed':spec['baseline_noise_seed'],'random_policy_seed':spec['random_policy_seed'],
                              'candidate_noise_seed_base':spec['candidate_noise_seeds']['downward']-101})
        core.write(trial/'protocol.json',child);core.write(trial/'environment.json',env)
        baseline=trial/'baseline';baseline.mkdir()
        core.write(baseline/'observations.json',[core.serialize(core.generate(child,0.,spec['baseline_noise_seed']))])
        core.trace(trial,'hypothesis','Compare observation-derived adaptive information choice with uniform random and fixed downward; truth remains evaluator-only.',[trial/'protocol.json',baseline/'observations.json'])
    core.trace(root,'hypothesis','Vary six physical settings, each with two noise replicates, to test whether adaptive selection differs from a fixed useful input.',[root/'protocol.json',root/'environment.json'])
    excluded={'initial-manifest.json','thoughts.jsonl'}|{f"trials/{s['trial_id']}/thoughts.jsonl" for s in protocol['trial_specs']}
    core.write(root/'initial-manifest.json',{'artifacts_sha256':core.inventory(root,excluded)})
    command=[sys.executable,str(sources/SELF),'run-trials','--campaign',str(root)]
    core.write(root/'invocation.json',{'argv':command,'started_at_utc':core.now()})
    try:subprocess.run(command,check=True)
    except Exception:
        (root/'controller-stack.txt').write_text(traceback.format_exc());raise


def choose(context,random_seed):
    core.validate_context(context)
    adaptive=max(context['candidates'],key=lambda item:item['predicted_information_logdet'])['id']
    return {'adaptive_numerical':adaptive,'uniform_random':core.uniform_random_candidate(random_seed),'fixed_downward':'downward'}


def seal_trial(trial,complete):
    core.write(trial/'manifest.json',{'status':'complete' if complete else 'completed_with_failures',
               'compute_elapsed_s':sum(core.read(p)['elapsed_s'] for p in trial.glob('*/process.json')),
               'artifacts_sha256':core.inventory(trial,{'manifest.json'})})
    core.verify(trial)


def run_trial(trial):
    protocol=core.read(trial/'protocol.json');spec=protocol['trial_spec']
    baseline=trial/'baseline'
    status=core.run_job(trial,'baseline')
    core.trace(trial,'evidence','Baseline fit retains observation-derived diagnostics and separate evaluator results.',list(baseline.iterdir()))
    if status['status']!='worker_finished' or not (baseline/'result.json').exists():
        seal_trial(trial,False);return
    context=core.selector_context(core.read(baseline/'result.json'),protocol)
    core.write(trial/'context.json',context)
    decisions=choose(context,spec['random_policy_seed'])
    # These are numerical/random/fixed policy records, never fabricated model responses.
    core.write(trial/'decisions.json',{'policies':decisions,'frozen_at_utc':core.now(),
               'rationale':{'adaptive_numerical':'Frozen maximum local information logdet; tie break listed order.',
                            'uniform_random':'Independent frozen seeded uniform draw, no tuning.',
                            'fixed_downward':'Always downward, regardless of observations.'}})
    core.trace(trial,'decision','Freeze all three explicitly identified policies before any new measurements.',[trial/'context.json',trial/'decisions.json'])
    for policy,candidate in decisions.items():
        job=trial/policy;job.mkdir()
        acceleration=next(a for identifier,a,_ in core.CANDIDATES if identifier==candidate)
        new=core.serialize(core.generate(protocol,acceleration,spec['candidate_noise_seeds'][candidate]))
        core.write(job/'new-observations.json',new)
        core.write(job/'observations.json',core.read(baseline/'observations.json')+[new])
        core.trace(trial,'action',f'{policy} chose {candidate}; candidate-keyed noise and baseline initialization are shared for matching choices.',[job/'observations.json',job/'new-observations.json',baseline/'checkpoint.pt'])
        core.run_job(trial,policy)
        core.trace(trial,'evidence',f'{policy} independently trained with equal optimizer limits under the non-borrowable trial budget.',list(job.iterdir()))
    complete=all((trial/name/'result.json').exists() and core.read(trial/name/'process.json')['status']=='worker_finished' for name in ['baseline',*POLICIES])
    seal_trial(trial,complete)


def collect(root):
    rows=[]
    for spec in core.read(root/'protocol.json')['trial_specs']:
        trial=root/'trials'/spec['trial_id'];row={**spec,'complete':core.read(trial/'manifest.json')['status']=='complete','fits':{}}
        decisions=core.read(trial/'decisions.json')['policies'] if (trial/'decisions.json').exists() else {}
        for name in ['baseline',*POLICIES]:
            path=trial/name/'result.json';process=trial/name/'process.json'
            result=core.read(path) if path.exists() else {'status':'unavailable','process':core.read(process) if process.exists() else None}
            result['candidate_id']='neutral' if name=='baseline' else decisions.get(name)
            row['fits'][name]=result
        rows.append(row)
    return rows


def summarize(rows):
    if len(rows)!=12 or not all(row['complete'] for row in rows):
        return {'status':'unavailable_incomplete','means':None,'paired_differences':None,'setting_means':None,
                'reason':'At least one trial is missing or failed; no successful-subset aggregate ranking.'}
    names=['baseline',*POLICIES]
    means={name:float(np.mean([row['fits'][name]['evaluation'][PRIMARY] for row in rows])) for name in names}
    pairs={comparison:[row['fits']['adaptive_numerical']['evaluation'][PRIMARY]-row['fits'][comparison]['evaluation'][PRIMARY] for row in rows]
           for comparison in ['fixed_downward','uniform_random']}
    settings={}
    for setting in sorted({row['setting_id'] for row in rows}):
        group=[row for row in rows if row['setting_id']==setting]
        if len(group)!=2:raise ValueError('each physical setting needs exactly two noise replicates')
        settings[setting]={'replicates':2,'means':{name:float(np.mean([r['fits'][name]['evaluation'][PRIMARY] for r in group])) for name in names},
                           'paired_means':{comparison:float(np.mean([r['fits']['adaptive_numerical']['evaluation'][PRIMARY]-r['fits'][comparison]['evaluation'][PRIMARY] for r in group])) for comparison in pairs}}
    matches=sum(row['fits']['adaptive_numerical']['candidate_id']=='downward' for row in rows)
    return {'status':'complete_descriptive_only','means':means,'paired_differences':pairs,'setting_means':settings,
            'adaptive_matches_fixed_downward':matches,'trials':12,'physical_settings':6,
            'conclusion':'Adaptive selected downward in every tested case: no evidence adaptation helps over the fixed input within this tested range.' if matches==12 else 'Adaptive choices differed from fixed downward; inspect descriptive paired differences without claiming broad superiority.',
            'interpretation':'Two noise replicates per physical-setting cluster; do not treat 12 trials as 12 independent physical settings.'}


def identity_checks(root,rows):
    checks=[]
    for row in rows:
        trial=root/'trials'/row['trial_id']
        if not row['complete']:
            checks.append({'trial_id':row['trial_id'],'available':False});continue
        starts=[core.read(trial/p/'initial-state.json') for p in POLICIES]
        pairs=[]
        for index,a in enumerate(POLICIES):
            for b in POLICIES[index+1:]:
                if row['fits'][a]['candidate_id']==row['fits'][b]['candidate_id']:
                    pairs.append({'policies':[a,b],'observations_identical':(trial/a/'observations.json').read_bytes()==(trial/b/'observations.json').read_bytes(),
                                  'checkpoint_bytes_identical':(trial/a/'checkpoint.pt').read_bytes()==(trial/b/'checkpoint.pt').read_bytes()})
        checks.append({'trial_id':row['trial_id'],'available':True,'starting_fingerprints_identical':all(s==starts[0] for s in starts),
                       'matched_policy_pairs':pairs})
    return checks


def report(root,rows,summary,checks):
    lines=['# Physical-setting adaptation test','',
           'Six physical settings (floor 0.03/0.12/0.22m crossed with restitution 0.65/0.90), two fresh noise replicates each. Model initialization is fixed. Policies are adaptive numerical information, independent uniform random, and fixed downward. No LLM choices are generated.', '',
           'Primary metric is analytic parameter-transfer position RMSE on reserved inputs, NOT direct neural generalization.', '',
           '|Setting|Replicate|Fit|Choice|Transfer RMSE|Timing MAE|Floor error|Restitution error|Neural reconstruction RMSE|',
           '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        for name,fit in row['fits'].items():
            e=fit.get('evaluation')
            if e is None:lines.append(f"|{row['setting_id']}|{row['noise_replicate']}|{name}|{fit['candidate_id']}|unavailable|—|—|—|—|");continue
            lines.append(f"|{row['setting_id']}|{row['noise_replicate']}|{name}|{fit['candidate_id']}|{e[PRIMARY]:.6f}|{e['parameter_transfer_impact_time_mae_s']:.6f}|{e['floor_absolute_error_m']:.6f}|{e['restitution_absolute_error']:.6f}|{e['direct_neural_reconstruction_position_rmse_m']:.6f}|")
    lines+=['',f"Aggregate status: {summary['status']}."]
    if summary['means'] is None:lines+=['',summary['reason']]
    else:
        lines+=['',summary['conclusion'],'',f"Adaptive/fixed choice agreement: {summary['adaptive_matches_fixed_downward']}/12.", '',
                '|Physical setting|Baseline mean|Adaptive mean|Random mean|Fixed-downward mean|','|---|---:|---:|---:|---:|']
        for setting,item in summary['setting_means'].items():
            m=item['means'];lines.append(f"|{setting}|{m['baseline']:.6f}|{m['adaptive_numerical']:.6f}|{m['uniform_random']:.6f}|{m['fixed_downward']:.6f}|")
        lines+=['','|Comparison: adaptive minus comparator|Mean paired difference|Minimum|Maximum|','|---|---:|---:|---:|']
        for comparator,values in summary['paired_differences'].items():
            lines.append(f'|{comparator}|{np.mean(values):.6f}|{min(values):.6f}|{max(values):.6f}|')
    lines+=['','All 12 paired differences, six setting-cluster means, full flow/guard/reset/event diagnostics, actual optimizer closure calls, elapsed times, classical comparators and identity checks are in results.json and each trial result. Matched choices use identical measurements and learned starting states but independent training jobs.', '',
            'Every prior/candidate/evaluation horizon was validated for one impact; second impacts and Zeno continuation are excluded. This is a descriptive six-setting development experiment, not 12 independent physical settings. No tuning, seed balancing, retries, missing-trial removal or successful-subset ranking. Token/model usage unavailable from collaboration is null.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run_trials(root):
    source_check(root);core.check_manifest(root,'initial-manifest.json')
    for spec in core.read(root/'protocol.json')['trial_specs']:
        trial=root/'trials'/spec['trial_id'];run_trial(trial)
        core.trace(root,'evidence',f"{spec['trial_id']} retained at its frozen physical setting and 50-second allowance.",[trial/'manifest.json'])
        print(f"FINISHED={spec['trial_id']}",flush=True)
    rows=collect(root);summary=summarize(rows);checks=identity_checks(root,rows)
    core.write(root/'results.json',{'trials':rows,'aggregate':summary,'identity_checks':checks})
    report(root,rows,summary,checks)
    core.trace(root,'next_step','Compare adaptation directly with fixed downward before planning further selector complexity.',[root/'results.json',root/'report.md'])
    processes=list((root/'trials').glob('*/*/process.json'))
    counts={status:sum(core.read(p)['status']==status for p in processes) for status in sorted({core.read(p)['status'] for p in processes})}
    core.write(root/'manifest.json',{'status':'complete' if summary['means'] is not None else 'completed_with_failures',
               'compute_elapsed_s':sum(core.read(p)['elapsed_s'] for p in processes),'training_process_status_counts':counts,
               'artifacts_sha256':core.inventory(root,{'manifest.json'})})
    verify(root)
    print(f'COMPLETE={root}',flush=True)


def verify(root):
    source_check(root)
    manifest=core.check_manifest(root,'manifest.json')
    if set(manifest['artifacts_sha256'])!=set(core.inventory(root,{'manifest.json'})):raise ValueError('sweep artifact inventory mismatch')
    for row in map(json.loads,(root/'thoughts.jsonl').read_text().splitlines()):
        for name,digest in row['artifacts_sha256'].items():
            if manifest['artifacts_sha256'].get(name)!=digest:raise ValueError('sweep trace mismatch')
    specs=core.read(root/'protocol.json')['trial_specs']
    for spec in specs:
        trial=root/'trials'/spec['trial_id'];core.verify(trial)
        if core.read(trial/'manifest.json')['compute_elapsed_s']>50 or len(list(trial.glob('*/invocation.json')))>4:
            raise ValueError('per-trial budget violation')
    if manifest['compute_elapsed_s']>600 or sum(manifest['training_process_status_counts'].values())>48:
        raise ValueError('sweep total budget violation')
    for check in core.read(root/'results.json')['identity_checks']:
        if check['available'] and (not check['starting_fingerprints_identical'] or any(not pair['observations_identical'] or not pair['checkpoint_bytes_identical'] for pair in check['matched_policy_pairs'])):
            raise ValueError('matched-policy fairness/identity failure')
    print('Physical sweep sources, artifacts, traces, budgets and identities verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','run-trials','verify']);parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path)
    args=parser.parse_args()
    if args.command=='run':run(args.out.resolve())
    elif args.command=='run-trials':run_trials(args.campaign.resolve())
    else:verify(args.campaign.resolve())


if __name__=='__main__':main()
