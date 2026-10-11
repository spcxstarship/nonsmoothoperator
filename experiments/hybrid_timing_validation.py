"""Frozen conditional timing validation with two fixed-direction comparators.

prepare --out NEW freezes10 worlds/schedules/streams before baseline training.
finish --campaign DIR --decision-map JSON validates all actual fresh-agent responses.
Execute finish/verify with DIR/sources/experiments/hybrid_timing_validation.py.
Eight main late-window trials; original/early controls never enter main averages.
Maximum60 real training jobs,60s pertrial without borrowing,600s total. No tuning.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import hybrid_selection_round as core
import hybrid_physical_sweep as physical

SELF='experiments/hybrid_timing_validation.py'
POLICIES=['adaptive_numerical','fresh_agent','fixed_downward','fixed_upward','uniform_random']
PRIMARY='parameter_transfer_position_rmse_m'


def design():
    rows=[]
    for floor in [.04,.21]:
        for restitution in [.68,.88]:
            for replicate in [1,2]:
                rows.append({'kind':'main','setting_id':f'floor-{floor:.2f}-e-{restitution:.2f}','floor_m':floor,
                             'restitution':restitution,'noise_replicate':replicate,'schedule':'late-0.56',
                             'times_s':np.linspace(.56,.7,12).tolist()})
    rows += [{'kind':'control','setting_id':'original-control','floor_m':.14,'restitution':.82,'noise_replicate':1,
              'schedule':'original','times_s':core.TIMES},
             {'kind':'control','setting_id':'early-rank-deficient-control','floor_m':.14,'restitution':.82,'noise_replicate':1,
              'schedule':'early-parent-0.35','times_s':np.r_[np.linspace(.05,.35,11),.7].tolist()}]
    for index,row in enumerate(rows):
        row.update({'trial_id':f'trial-{index+1:02d}','baseline_noise_seed':700001+index,'random_policy_seed':1100000+index,
                    'candidate_noise_seeds':{identifier:900000+1000*index+offset for identifier,_,offset in core.CANDIDATES}})
    return rows


def overrides(spec):
    return {'extra_source_names':['experiments/hybrid_physical_sweep.py',SELF],
            'simulator':{'floor_m':spec['floor_m'],'restitution':spec['restitution']},
            'data':{'times_s':spec['times_s'],'baseline_seed':spec['baseline_noise_seed'],
                    'random_policy_seed':spec['random_policy_seed'],
                    'candidate_noise_seed_base':spec['candidate_noise_seeds']['downward']-101},
            'budgets':{'maximum_training_jobs':6,'job_timeout_s':120,'total_compute_s':60,'cleanup_reserve_s':.5}}


def source_check(root):
    for name,digest in core.read(root/'protocol.json')['source_sha256'].items():
        if core.sha(root/'sources'/name)!=digest:raise ValueError(f'final validation source mismatch: {name}')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    names=core.SOURCE_NAMES+['experiments/hybrid_physical_sweep.py',SELF]
    sources=root/'sources'
    for name in names:
        target=sources/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    common=physical.common();common['budgets'].update({'maximum_training_jobs':6,'total_compute_s':60})
    for spec in design():
        for identifier,u,_ in core.CANDIDATES:
            obs=core.Observations(1.,u,np.array(spec['times_s']),np.zeros(12))
            core.validate_observations([obs],common['inference']['floor_bounds_m'],common['inference']['restitution_bounds'],9.81)
            core.predict(spec['floor_m'],spec['restitution'],1.,u,np.array(spec['times_s']))
    protocol={'study':'conditional validation following exploratory timing audit; NOT global confirmation',
              'frozen_at_utc':core.now(),'trial_specs':design(),'common':common,'policies':POLICIES,
              'maximum_training_jobs':60,'maximum_total_compute_s':600,'per_trial_compute_s':60,'budget_borrowing':False,
              'model_seed':0,'source_sha256':{name:core.sha(sources/name) for name in names},
              'analysis':{'main':'Eight late-window trials at four physical-setting clusters, two noise replicates each.',
                          'controls':'Two explicitly separate original/early schedule controls, never included in main means.',
                          'comparators':'Both fixed downward and fixed upward were declared before outcomes, plus uniform random.',
                          'statistics':'All8 paired differences and per-setting means, descriptive only; no hypothesis tests or bootstrap.',
                          'failure_rule':'Any missing/failed fit across10 trials suppresses aggregate ranking; retain all outcomes, no retries/tuning.'},
              'selector_limitations':'Information scores are local plug-in predictions from uncalibrated point estimates. Early control has rank-deficient baseline information. Score-informed agent agreement does not establish intelligence gain.',
              'random_rule':'Frozen independent numpy.default_rng(seed).integers(0,3), listed candidate order; no balancing.'}
    core.write(root/'protocol.json',protocol)
    core.trace(root,'hypothesis','A full-rank late sensor window may make informative actuation depend on learned floor; compare both fixed directions and retain original/early controls separately.',[root/'protocol.json'])
    command=[sys.executable,str(sources/SELF),'prepare-trials','--campaign',str(root)]
    core.write(root/'prepare-invocation.json',{'argv':command,'started_at_utc':core.now()})
    try:subprocess.run(command,check=True)
    except Exception:
        (root/'prepare-controller-stack.txt').write_text(traceback.format_exc());raise


def prepare_trials(root):
    source_check(root);trials=root/'trials';trials.mkdir();statuses=[]
    specs=core.read(root/'protocol.json')['trial_specs']
    for spec in specs:
        core.prepare(trials/spec['trial_id'],overrides(spec),defer_training=True)
    excluded={'input-manifest.json','thoughts.jsonl'}|{f"trials/{s['trial_id']}/thoughts.jsonl" for s in specs}
    core.write(root/'input-manifest.json',{'artifacts_sha256':core.inventory(root,excluded)})
    core.check_manifest(root,'input-manifest.json')
    for spec in specs:
        target=trials/spec['trial_id'];core.complete_prepare(target)
        ready=(target/'prepare-manifest.json').exists()
        if ready:
            core.write(target/'fixed-decisions.json',{'fixed_downward':'downward','fixed_upward':'upward'})
        statuses.append({'trial_id':spec['trial_id'],'kind':spec['kind'],'status':'awaiting_selector' if ready else 'baseline_failed',
                         'context':str(target/'selector/ball/trial/round/context.json') if ready else None,
                         'instructions':str(target/'selector/ball/trial/round/instructions.txt') if ready else None})
        core.trace(root,'evidence',f"{spec['trial_id']} baseline retained; fresh agents receive only frozen observation-derived context and instructions.",[target/'protocol.json',target/'baseline/process.json'])
    core.write(root/'prepared-trials.json',statuses)
    excluded={'prepare-manifest.json','thoughts.jsonl'}
    for spec in core.read(root/'protocol.json')['trial_specs']:
        prefix=f"trials/{spec['trial_id']}/";excluded.update({prefix+'thoughts.jsonl',prefix+'selector/ball/trial/round/manifest.json'})
    core.write(root/'prepare-manifest.json',{'artifacts_sha256':core.inventory(root,excluded)})
    verify(root);print(f'PREPARED={root/"prepared-trials.json"}',flush=True)


def validate_choices(root,map_path):
    entries=core.read(map_path);ready=[r['trial_id'] for r in core.read(root/'prepared-trials.json') if r['status']=='awaiting_selector']
    if not isinstance(entries,list) or len(entries)!=len(ready) or {e['trial_id'] for e in entries}!=set(ready):
        raise ValueError('decision map must cover every successful baseline exactly once')
    by_id={e['trial_id']:e for e in entries}
    for trial_id,item in by_id.items():
        target=root/'trials'/trial_id;core.verify_prepared(target)
        context=core.read(target/'selector/ball/trial/round/context.json')
        core.parse_decision(Path(item['decision']).read_text(),core.validate_context(context))
        if not isinstance(core.read(item['metadata']),dict):raise ValueError('metadata must be a JSON object')
    return by_id


def finish_trial(trial,decision_path,metadata_path):
    core.source_check(trial);core.verify_prepared(trial)
    context=core.read(trial/'selector/ball/trial/round/context.json')
    choice=core.parse_decision(decision_path.read_text(),core.validate_context(context))
    core.write(trial/'finish-lock.json',{'started_at_utc':core.now()})
    shutil.copyfile(decision_path,trial/'selector-response.json');shutil.copyfile(metadata_path,trial/'selector-task-metadata.json')
    decisions={'adaptive_numerical':core.read(trial/'numerical-decision.json')['candidate_id'],'fresh_agent':choice.candidate_id,
               **core.read(trial/'fixed-decisions.json'),'uniform_random':core.read(trial/'random-decision.json')['candidate_id']}
    core.write(trial/'all-decisions.json',{'policies':decisions,'frozen_at_utc':core.now()})
    core.write(trial/'usage.json',{'source':'collaboration API does not expose token usage','elapsed_ms':None,'input_tokens':None,
                                 'cached_input_tokens':None,'output_tokens':None,'reasoning_tokens':None})
    core.trace(trial,'decision','Actual fresh-agent response and all five policy decisions frozen before new evaluator measurements.',[trial/'selector-response.json',trial/'selector-task-metadata.json',trial/'all-decisions.json'])
    protocol=core.read(trial/'protocol.json');baseline=trial/'baseline'
    for policy,candidate in decisions.items():
        job=trial/policy;job.mkdir();u=next(u for identifier,u,_ in core.CANDIDATES if identifier==candidate)
        seed=protocol['data']['candidate_noise_seed_base']+protocol['data']['candidate_seed_offsets'][candidate]
        new=core.serialize(core.generate(protocol,u,seed))
        core.write(job/'new-observations.json',new);core.write(job/'observations.json',core.read(baseline/'observations.json')+[new])
        core.trace(trial,'action',f'{policy} selected {candidate}; actual timing, candidate-keyed measurements and copied baseline state determine the update.',[job/'observations.json',baseline/'checkpoint.pt'])
        core.run_job(trial,policy)
        core.trace(trial,'evidence',f'{policy} independently trained under equal optimizer limits and fixed trial compute allowance.',list(job.iterdir()))
    policies={name:{'candidate_id':candidate,'result':core.read(trial/name/'result.json') if (trial/name/'result.json').exists() else core.read(trial/name/'process.json')} for name,candidate in decisions.items()}
    result={'candidate_id':choice.candidate_id,'policies':policies,'completed_at_utc':core.now()};core.write(trial/'round-result.json',result)
    core.round_store.finish(argparse.Namespace(directory=trial/'selector/ball/trial/round',decision=trial/'selector-response.json',result=trial/'round-result.json',usage=trial/'usage.json'))
    complete=all((trial/name/'result.json').exists() and core.read(trial/name/'process.json')['status']=='worker_finished' for name in ['baseline',*POLICIES])
    physical.seal_trial(trial,complete)


def collect(root):
    rows=[]
    for spec in core.read(root/'protocol.json')['trial_specs']:
        trial=root/'trials'/spec['trial_id'];row={**spec,'complete':core.read(trial/'manifest.json')['status']=='complete' if (trial/'manifest.json').exists() else False,'fits':{}}
        decisions=core.read(trial/'all-decisions.json')['policies'] if (trial/'all-decisions.json').exists() else {}
        for name in ['baseline',*POLICIES]:
            p=trial/name/'result.json';process=trial/name/'process.json'
            fit=core.read(p) if p.exists() else {'status':'unavailable','process':core.read(process) if process.exists() else None}
            fit['candidate_id']='neutral' if name=='baseline' else decisions.get(name);row['fits'][name]=fit
        context_path=trial/'selector/ball/trial/round/context.json'
        row['baseline_information']=core.read(context_path)['diagnostics'] if context_path.exists() else None
        rows.append(row)
    return rows


def summarize(rows):
    if len(rows)!=10 or not all(r['complete'] for r in rows):
        return {'status':'unavailable_incomplete','means':None,'paired_differences':None,'setting_means':None,
                'reason':'Missing/failed trial: all aggregate rankings suppressed, no successful-subset conclusion.'}
    main=[r for r in rows if r['kind']=='main'];controls=[r for r in rows if r['kind']=='control']
    if len(main)!=8 or len(controls)!=2:raise ValueError('exactly8main and2separate controls required')
    means={name:float(np.mean([r['fits'][name]['evaluation'][PRIMARY] for r in main])) for name in ['baseline',*POLICIES]}
    classical_means={name:float(np.mean([r['fits'][name]['classical']['evaluation'][PRIMARY] for r in main])) for name in ['baseline',*POLICIES]}
    paired={policy:{comparator:[r['fits'][policy]['evaluation'][PRIMARY]-r['fits'][comparator]['evaluation'][PRIMARY] for r in main]
                    for comparator in ['fixed_downward','fixed_upward','uniform_random']} for policy in ['adaptive_numerical','fresh_agent']}
    settings={setting:{name:float(np.mean([r['fits'][name]['evaluation'][PRIMARY] for r in main if r['setting_id']==setting])) for name in ['baseline',*POLICIES]}
              for setting in sorted({r['setting_id'] for r in main})}
    agreement=sum(r['fits']['adaptive_numerical']['candidate_id']==r['fits']['fresh_agent']['candidate_id'] for r in main)
    return {'status':'complete_descriptive_only','main_trials':8,'physical_setting_clusters':4,'controls_excluded':2,
            'means':means,'classical_means':classical_means,'paired_differences':paired,'setting_means':settings,
            'agent_numerical_choice_agreement':agreement,
            'choice_counts':{p:{c:sum(r['fits'][p]['candidate_id']==c for r in main) for c,_,_ in core.CANDIDATES} for p in POLICIES},
            'interpretation':'Conditional development validation after timing exploration, not confirmation. Both fixed directions were declared before outcomes. Local plug-in information is not calibrated uncertainty; score-informed agent agreement is not intelligence gain.'}


def identity_checks(root,rows):
    checks=[]
    for row in rows:
        trial=root/'trials'/row['trial_id']
        if not row['complete']:checks.append({'trial_id':row['trial_id'],'available':False});continue
        starts=[core.read(trial/p/'initial-state.json') for p in POLICIES]
        pairs=[]
        for index,a in enumerate(POLICIES):
            for b in POLICIES[index+1:]:
                if row['fits'][a]['candidate_id']==row['fits'][b]['candidate_id']:
                    pairs.append({'policies':[a,b],'observations_identical':(trial/a/'observations.json').read_bytes()==(trial/b/'observations.json').read_bytes(),
                                  'checkpoint_bytes_identical':(trial/a/'checkpoint.pt').read_bytes()==(trial/b/'checkpoint.pt').read_bytes()})
        checks.append({'trial_id':row['trial_id'],'available':True,'starting_fingerprints_identical':all(s==starts[0] for s in starts),'matched_policy_pairs':pairs})
    return checks


def report(root,rows,summary):
    lines=['# Conditional timing validation','',
           'MAIN: eight late-window trials at four new physical settings, two fresh noise replicates each. Original and early timing controls are reported separately. No repeated design tuning or favorable-outcome case selection.', '',
           'Primary metric is analytic parameter-transfer position RMSE on fixed reserved inputs, NOT direct neural generalization. Actual measurement schedules are used throughout. Both fixed directions are necessary comparators; information scores are uncalibrated point-estimate predictions.', '',
           '|Trial/type|Fit|Choice|PINN transfer RMSE|Classical transfer RMSE|Timing MAE|Floor error|Restitution error|Neural reconstruction RMSE|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for name,fit in row['fits'].items():
            e=fit.get('evaluation')
            if e is None:lines.append(f"|{row['trial_id']}/{row['kind']}|{name}|{fit['candidate_id']}|unavailable|—|—|—|—|—|");continue
            ce=fit['classical']['evaluation'][PRIMARY]
            lines.append(f"|{row['trial_id']}/{row['kind']}|{name}|{fit['candidate_id']}|{e[PRIMARY]:.6f}|{ce:.6f}|{e['parameter_transfer_impact_time_mae_s']:.6f}|{e['floor_absolute_error_m']:.6f}|{e['restitution_absolute_error']:.6f}|{e['direct_neural_reconstruction_position_rmse_m']:.6f}|")
    lines+=['',f"Aggregate status: {summary['status']}."]
    if summary['means'] is None:lines+=['',summary['reason']]
    else:
        lines+=['','|Main fit|Mean PINN transfer RMSE|Mean classical transfer RMSE|','|---|---:|---:|']
        for name,value in summary['means'].items():lines.append(f"|{name}|{value:.6f}|{summary['classical_means'][name]:.6f}|")
        lines+=['',f"Fresh-agent/numerical choice agreement: {summary['agent_numerical_choice_agreement']}/8 main cases. Agreement with exposed scores does not establish intelligence gain.", '',
                '|Main paired difference: policy−comparator|Mean|Minimum|Maximum|','|---|---:|---:|---:|']
        for policy,comparisons in summary['paired_differences'].items():
            for comparator,values in comparisons.items():lines.append(f'|{policy}−{comparator}|{np.mean(values):.6f}|{min(values):.6f}|{max(values):.6f}|')
        lines+=['','Per-world means, all 8 paired differences and actual choice counts are retained in results.json. Controls never enter these averages.']
    lines+=['','Control information diagnostics:']
    for row in rows:
        if row['kind']=='control' and row['baseline_information']:
            d=row['baseline_information'];lines.append(f"- {row['setting_id']}: unregularized initial Fisher rank {d['initial_information_rank']}; condition {d['initial_information_condition_number']}; eigenvalues {d['initial_information_eigenvalues']}.")
    lines+=['','The early control has a rank-deficient two-parameter baseline: one post-impact observation does not identify both floor and restitution. Neural regularization/optimization yields an uncalibrated point estimate; classical starts may expose nonidentifiability. No posterior calibration is claimed.', '',
            'Full flow/guard/reset/event diagnostics, closure calls, checkpoints, exact agent responses/task metadata, input/source hashes, errors and retained failures are in trial artifacts. All policies have24 observations, copied baseline weights/physical parameters, identical new-module initialization and equal optimizer budgets. Matching choices share new noise but still run separately.', '',
            'This bounded synthetic study is conditional on timing exploration and restricted to known one-impact dynamics/prior. It does not establish real-world performance, global policy superiority, or novelty. A negative or incomplete result is retained without retuning.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def finish(root,map_path):
    source_check(root);core.check_manifest(root,'prepare-manifest.json');by_id=validate_choices(root,map_path)
    core.write(root/'finish-lock.json',{'started_at_utc':core.now()});incoming=root/'incoming';incoming.mkdir()
    shutil.copyfile(map_path,root/'decision-map.json')
    for trial_id,item in by_id.items():
        target=incoming/trial_id;target.mkdir();shutil.copyfile(item['decision'],target/'decision.json');shutil.copyfile(item['metadata'],target/'metadata.json')
    core.write(root/'selection-manifest.json',{'artifacts_sha256':core.inventory(incoming,set())})
    core.trace(root,'decision','All actual fresh-agent responses preserved collectively before any new measurements or policy training.',[root/'decision-map.json',root/'selection-manifest.json'])
    for spec in core.read(root/'protocol.json')['trial_specs']:
        trial_id=spec['trial_id'];trial=root/'trials'/trial_id
        if trial_id not in by_id:continue
        finish_trial(trial,incoming/trial_id/'decision.json',incoming/trial_id/'metadata.json')
        core.trace(root,'evidence',f'{trial_id} retained five independent updates under its 60-second allowance.',[trial/'manifest.json'])
        print(f'FINISHED={trial_id}',flush=True)
    rows=collect(root);summary=summarize(rows);checks=identity_checks(root,rows)
    core.write(root/'results.json',{'trials':rows,'aggregate':summary,'identity_checks':checks});report(root,rows,summary)
    core.trace(root,'next_step','Assess adaptive choices against BOTH fixed directions and classical inversion; do not convert local-score agreement into an intelligence claim.',[root/'results.json',root/'report.md'])
    processes=list((root/'trials').glob('*/*/process.json'))
    counts={s:sum(core.read(p)['status']==s for p in processes) for s in sorted({core.read(p)['status'] for p in processes})}
    core.write(root/'manifest.json',{'status':'complete' if summary['means'] is not None else 'completed_with_failures',
               'compute_elapsed_s':sum(core.read(p)['elapsed_s'] for p in processes),'training_process_status_counts':counts,
               'artifacts_sha256':core.inventory(root,{'manifest.json'})})
    verify(root);print(f'COMPLETE={root}',flush=True)


def verify(root):
    source_check(root)
    if not (root/'manifest.json').exists():
        core.check_manifest(root,'prepare-manifest.json')
        for item in core.read(root/'prepared-trials.json'):
            if item['status']=='awaiting_selector':core.verify_prepared(root/'trials'/item['trial_id'])
        print('Final-study prepared artifacts verified',flush=True);return
    manifest=core.check_manifest(root,'manifest.json')
    if set(manifest['artifacts_sha256'])!=set(core.inventory(root,{'manifest.json'})):raise ValueError('final study inventory mismatch')
    for row in map(json.loads,(root/'thoughts.jsonl').read_text().splitlines()):
        for name,digest in row['artifacts_sha256'].items():
            if manifest['artifacts_sha256'].get(name)!=digest:raise ValueError('final study trace mismatch')
    for spec in core.read(root/'protocol.json')['trial_specs']:
        trial=root/'trials'/spec['trial_id'];core.verify(trial)
        if core.read(trial/'manifest.json')['compute_elapsed_s']>60 or len(list(trial.glob('*/invocation.json')))>6:raise ValueError('trial budget exceeded')
    if manifest['compute_elapsed_s']>600 or sum(manifest['training_process_status_counts'].values())>60:raise ValueError('overall budget exceeded')
    for check in core.read(root/'results.json')['identity_checks']:
        if check['available'] and (not check['starting_fingerprints_identical'] or any(not p['observations_identical'] or not p['checkpoint_bytes_identical'] for p in check['matched_policy_pairs'])):
            raise ValueError('matched policy identity/fairness failure')
    print('Final study manifests, traces, budgets and matched identities verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','prepare-trials','finish','verify']);parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path);parser.add_argument('--decision-map',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out.resolve())
    elif args.command=='prepare-trials':prepare_trials(args.campaign.resolve())
    elif args.command=='finish':finish(args.campaign.resolve(),args.decision_map)
    else:verify(args.campaign.resolve())


if __name__=='__main__':main()
