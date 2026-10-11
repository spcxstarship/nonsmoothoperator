"""Six fixed noise datasets comparing fresh-agent, information, and random policies.

prepare --out NEW freezes every seed/budget/source before training six baselines.
finish --campaign DIR --decision-map JSON validates all fresh-agent choices first.
Execute finish/verify with DIR/sources/experiments/hybrid_selection_replication.py.
Decision map: [{"trial_id":"trial-1", "decision":"/path/raw.json", "metadata":"/path/task.json"}, ...].
No retries, tuning, successful-subset ranking, paid APIs, or publication.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import hybrid_selection_round as round_impl

SELF='experiments/hybrid_selection_replication.py'
POLICIES=['llm','numerical','random']
PRIMARY='parameter_transfer_position_rmse_m'


def seed_schedule():
    return [{'trial_id':f'trial-{index+1}','index_zero_based':index,
             'baseline_noise_seed':1001+index,'random_policy_seed':20261007+index,
             'candidate_noise_seed_base':50000+1000*index,
             'candidate_noise_seeds':{identifier:50000+1000*index+offset for identifier,_,offset in round_impl.CANDIDATES}}
            for index in range(6)]


def override(spec):
    return {'extra_source_names':[SELF],
            'data':{'baseline_seed':spec['baseline_noise_seed'],'random_policy_seed':spec['random_policy_seed'],
                    'candidate_noise_seed_base':spec['candidate_noise_seed_base']},
            'budgets':{'maximum_training_jobs':4,'job_timeout_s':120,'total_compute_s':100,'cleanup_reserve_s':.5}}


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    sources=root/'sources'
    names=round_impl.SOURCE_NAMES+[SELF]
    for name in names:
        target=sources/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    protocol={'study':'six fixed-noise synthetic development replications','frozen_at_utc':round_impl.now(),
              'trial_specs':seed_schedule(),'model_seed':0,'random_algorithm':'numpy.default_rng(seed).integers(0,3), uniform listed candidate order',
              'policies':POLICIES,'maximum_training_jobs':24,'per_trial_training_jobs':4,
              'job_timeout_s':120,'per_trial_compute_s':100,'overall_compute_s':600,'budget_borrowing':False,
              'simulator':{'floor_m':.12,'restitution':.8,'gravity_m_s2':9.81},
              'inference':{'floor_bounds_m':[0.,.25],'restitution_bounds':[.6,.95],'initial_floor_m':.05,'initial_restitution':.65},
              'pinn':{'width':16,'collocation_points_per_segment':48,'adam_steps':1500,'adam_learning_rate':.003,
                      'lbfgs_steps':1000,'physics_weight':100.,'guard_weight':100.},
              'observations':{'initial_height_m':1.,'times_s':round_impl.TIMES,'noise_std_m':.01,
                              'initial_acceleration_m_s2':0.,'candidate_accelerations_m_s2':[-4.,0.,4.],
                              'initial_count':12,'updated_count':24,'sharing':'same trial/candidate means identical new evaluator observations across policies'},
              'evaluation':{'initial_heights_m':[.85,1.15],'accelerations_m_s2':[-2.,2.],'horizon_s':.62,'points_per_trajectory':301,
                            'primary':PRIMARY,'definition':'Analytic parameter transfer, not direct neural generalization'},
              'information_rule':'Maximize logdet(F_initial+F_candidate+1e-9 I), finite-difference step1e-5 in dimensionless prior coordinates, noise scaled; listed-order tie break.',
              'analysis':{'means':'all six trials only','paired_difference':'policy minus random primary error on each same-noise trial',
                          'summary':'descriptive mean and range; no hypothesis tests or bootstrap',
                          'failure_rule':'retain every trial; any missing/failed baseline or policy fit suppresses all aggregate ranking/paired summaries, no successful-subset conclusion'},
              'source_sha256':{name:round_impl.sha(sources/name) for name in names}}
    round_impl.write(root/'protocol.json',protocol)
    round_impl.trace(root,'hypothesis','Replicate across six independently noisy datasets with model initialization fixed; compare paired policy updates without tuning or changing random choices.',[root/'protocol.json'])
    command=[sys.executable,str(sources/SELF),'prepare-trials','--campaign',str(root)]
    round_impl.write(root/'prepare-invocation.json',{'argv':command,'started_at_utc':round_impl.now()})
    try:
        subprocess.run(command,check=True)
    except Exception:
        (root/'prepare-controller-stack.txt').write_text(traceback.format_exc());raise


def prepare_trials(root):
    source_check(root)
    protocol=round_impl.read(root/'protocol.json')
    trials=root/'trials';trials.mkdir()
    statuses=[]
    for spec in protocol['trial_specs']:
        target=trials/spec['trial_id']
        round_impl.prepare(target,override(spec))
        prepared=(target/'prepare-manifest.json').exists()
        statuses.append({'trial_id':spec['trial_id'],'status':'awaiting_selector' if prepared else 'baseline_failed',
                         'context':str(target/'selector/ball/trial/round/context.json') if prepared else None,
                         'instructions':str(target/'selector/ball/trial/round/instructions.txt') if prepared else None})
        round_impl.trace(root,'evidence',f"{spec['trial_id']} baseline retained; selector files are ready only if its baseline succeeded.",[target/'protocol.json',target/'baseline/process.json'])
    round_impl.write(root/'prepared-trials.json',statuses)
    excluded={'prepare-manifest.json','thoughts.jsonl'}
    for spec in protocol['trial_specs']:
        prefix=f"trials/{spec['trial_id']}/"
        excluded.update({prefix+'thoughts.jsonl',prefix+'selector/ball/trial/round/manifest.json'})
    round_impl.write(root/'prepare-manifest.json',{'status':'awaiting_selectors','artifacts_sha256':round_impl.inventory(root,excluded)})
    verify(root)
    print(f'PREPARED_TRIALS={root/"prepared-trials.json"}',flush=True)


def source_check(root):
    for name,digest in round_impl.read(root/'protocol.json')['source_sha256'].items():
        if round_impl.sha(root/'sources'/name)!=digest:
            raise ValueError(f'replication source mismatch: {name}')


def validate_decisions(root,map_path):
    entries=round_impl.read(map_path)
    if not isinstance(entries,list):
        raise ValueError('decision map must be a list')
    ready=[item['trial_id'] for item in round_impl.read(root/'prepared-trials.json') if item['status']=='awaiting_selector']
    if len(entries)!=len(ready) or {item['trial_id'] for item in entries}!=set(ready):
        raise ValueError('decision map must cover every successful baseline exactly once')
    by_id={item['trial_id']:item for item in entries}
    for trial_id in ready:
        trial=root/'trials'/trial_id
        round_impl.verify_prepared(trial)
        item=by_id[trial_id]
        context=round_impl.read(trial/'selector/ball/trial/round/context.json')
        round_impl.parse_decision(Path(item['decision']).read_text(),round_impl.validate_context(context))
        if not isinstance(round_impl.read(item['metadata']),dict):
            raise ValueError('selector metadata must be a JSON object')
    return by_id


def aggregate(rows,expected_trials=6):
    complete=len(rows)==expected_trials and all(row['complete'] for row in rows)
    if not complete:
        return {'status':'unavailable_incomplete','reason':'At least one trial is missing or failed; no successful-subset aggregate ranking.',
                'means':None,'paired_vs_random':None}
    means={policy:float(np.mean([row['policies'][policy]['evaluation'][PRIMARY] for row in rows])) for policy in POLICIES}
    differences={}
    for policy in ['llm','numerical']:
        values=[row['policies'][policy]['evaluation'][PRIMARY]-row['policies']['random']['evaluation'][PRIMARY] for row in rows]
        differences[policy]={'per_trial':values,'mean':float(np.mean(values)),'minimum':min(values),'maximum':max(values),
                             'definition':'policy minus random; negative favors policy on this trial'}
    return {'status':'complete_descriptive_only','means':means,'paired_vs_random':differences,
            'interpretation':'n=6 noise datasets, fixed physical setting/model seed; local-score agreement is not intelligence advantage.'}


def collect(root):
    rows=[]
    for spec in round_impl.read(root/'protocol.json')['trial_specs']:
        trial=root/'trials'/spec['trial_id']; row={**spec,'baseline':None,'policies':{},'complete':True}
        for name in ['baseline',*POLICIES]:
            path=trial/name/'result.json'; process=trial/name/'process.json'
            successful=path.exists() and process.exists() and round_impl.read(process)['status']=='worker_finished'
            result=round_impl.read(path) if path.exists() else {'status':'unavailable','process':round_impl.read(process) if process.exists() else None}
            if not successful:row['complete']=False
            if name=='baseline':row['baseline']=result
            else:
                choice_path=trial/('selector-response.json' if name=='llm' else f'{name}-decision.json')
                result['candidate_id']=round_impl.read(choice_path)['candidate_id'] if choice_path.exists() else None
                row['policies'][name]=result
        rows.append(row)
    return rows


def report(root,rows,summary):
    lines=['# Six noise-replicated hybrid selection trials','',
           'Synthetic development comparison. Physical world and model initialization stay fixed; only measurement noise is replicated. Primary error is analytic parameter-transfer position RMSE on fixed reserved inputs, not direct neural generalization.', '',
           '|Trial|Fit|Choice|Transfer RMSE|Timing MAE|Floor error|Restitution error|Neural reconstruction RMSE|',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        for policy in ['baseline',*POLICIES]:
            result=row['baseline'] if policy=='baseline' else row['policies'][policy]
            e=result.get('evaluation');choice=result.get('candidate_id','neutral')
            if e is None:
                lines.append(f"|{row['trial_id']}|{policy}|{choice}|unavailable|—|—|—|—|")
            else:
                lines.append(f"|{row['trial_id']}|{policy}|{choice}|{e[PRIMARY]:.6f}|{e['parameter_transfer_impact_time_mae_s']:.6f}|{e['floor_absolute_error_m']:.6f}|{e['restitution_absolute_error']:.6f}|{e['direct_neural_reconstruction_position_rmse_m']:.6f}|")
    lines+=['',f"Aggregate status: {summary['status']}."]
    if summary['means'] is not None:
        lines+=['','|Policy|Mean primary RMSE|Mean paired difference versus random|Paired range|','|---|---:|---:|---|']
        for policy in POLICIES:
            diff=summary['paired_vs_random'].get(policy)
            lines.append(f"|{policy}|{summary['means'][policy]:.6f}|{diff['mean'] if diff else 0.:.6f}|{str([diff['minimum'],diff['maximum']]) if diff else 'reference'}|")
    else:lines+=['',summary['reason']]
    lines+=['','All decisions were frozen before additional measurements. Numerical scores were available equally to fresh isolated agents. Random actuation is an independent seeded uniform draw, without balancing or seed changes. Identical choices use identical new noisy observations and baseline warm starts; policy jobs remain separate.', '',
            'Detailed before/after flow, guard/reset diagnostics, observed event timing, L-BFGS closure calls, elapsed times, classical fits, exact decisions and failures are retained in trial result files and results.json. Learned physical parameters are separate from neural checkpoint weights. Failures remain in the ledger and suppress aggregate ranking.', '',
            'The entire prior supports one impact per trajectory within the observation horizon; second impacts and Zeno continuation are excluded. Six paired trials support descriptive differences only, not policy superiority or broad physical-setting generalization. Unknown token usage/model identity remains null.', '',
            'Prepare with `.venv/bin/python experiments/hybrid_selection_replication.py prepare --out data/research/NEW`. Finish/verify with the frozen runner in `CAMPAIGN/sources/experiments/hybrid_selection_replication.py`; provide `--campaign CAMPAIGN` and `--decision-map MAP` for finish.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def finish(root,map_path):
    source_check(root);round_impl.check_manifest(root,'prepare-manifest.json')
    by_id=validate_decisions(root,map_path)
    round_impl.write(root/'finish-lock.json',{'started_at_utc':round_impl.now()})
    incoming=root/'incoming';incoming.mkdir()
    shutil.copyfile(map_path,root/'decision-map.json')
    # Freeze all six exact responses before generating any new measurements.
    for trial_id,item in by_id.items():
        folder=incoming/trial_id;folder.mkdir()
        shutil.copyfile(item['decision'],folder/'decision.json');shutil.copyfile(item['metadata'],folder/'metadata.json')
    round_impl.write(root/'selection-manifest.json',{'artifacts_sha256':round_impl.inventory(incoming,set())})
    round_impl.trace(root,'decision','All available fresh-agent decisions and task metadata are preserved exactly before any policy update.',[root/'decision-map.json',root/'selection-manifest.json'])
    for spec in round_impl.read(root/'protocol.json')['trial_specs']:
        trial_id=spec['trial_id'];trial=root/'trials'/trial_id
        if trial_id not in by_id:continue
        round_impl.finish(trial,incoming/trial_id/'decision.json',incoming/trial_id/'metadata.json')
        round_impl.trace(root,'evidence',f'{trial_id} independent policy jobs retained under its fixed100-second allowance.',[trial/'manifest.json',trial/'round-result.json'])
    rows=collect(root);summary=aggregate(rows)
    round_impl.write(root/'results.json',{'trials':rows,'aggregate':summary})
    report(root,rows,summary)
    processes=list((root/'trials').glob('*/*/process.json'))
    elapsed=sum(round_impl.read(p)['elapsed_s'] for p in processes)
    counts={status:sum(round_impl.read(p)['status']==status for p in processes) for status in sorted({round_impl.read(p)['status'] for p in processes})}
    round_impl.trace(root,'next_step','Interpret paired differences as noise-replicated development evidence; check failures and policy agreement before planning further work.',[root/'results.json',root/'report.md'])
    round_impl.write(root/'manifest.json',{'status':'complete' if summary['means'] is not None else 'completed_with_failures',
                     'compute_elapsed_s':elapsed,'training_process_status_counts':counts,
                     'artifacts_sha256':round_impl.inventory(root,{'manifest.json'})})
    verify(root)
    print(f'COMPLETE={root} compute_seconds={elapsed:.3f}',flush=True)


def verify(root):
    source_check(root)
    if not (root/'manifest.json').exists():
        round_impl.check_manifest(root,'prepare-manifest.json')
        for item in round_impl.read(root/'prepared-trials.json'):
            if item['status']=='awaiting_selector':round_impl.verify_prepared(root/'trials'/item['trial_id'])
        print('Six-trial prepared frozen artifacts verified',flush=True);return
    manifest=round_impl.check_manifest(root,'manifest.json')
    if set(manifest['artifacts_sha256'])!=set(round_impl.inventory(root,{'manifest.json'})):
        raise ValueError('replication inventory mismatch')
    for row in map(json.loads,(root/'thoughts.jsonl').read_text().splitlines()):
        for name,digest in row['artifacts_sha256'].items():
            if manifest['artifacts_sha256'].get(name)!=digest:raise ValueError('replication trace mismatch')
    for spec in round_impl.read(root/'protocol.json')['trial_specs']:
        round_impl.verify(root/'trials'/spec['trial_id'])
    counts=manifest['training_process_status_counts']
    if sum(counts.values())>24 or manifest['compute_elapsed_s']>600:
        raise ValueError('replication budget integrity mismatch')
    print('Six-trial replication manifests and visible traces verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','prepare-trials','finish','verify'])
    parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path);parser.add_argument('--decision-map',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out.resolve())
    elif args.command=='prepare-trials':prepare_trials(args.campaign.resolve())
    elif args.command=='finish':finish(args.campaign.resolve(),args.decision_map)
    else:verify(args.campaign.resolve())


if __name__=='__main__':main()
