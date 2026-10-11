"""One bounded synthetic hybrid actuation-selection round, local artifacts only.

prepare --out NEW; finish --campaign DIR --decision JSON --metadata JSON;
verify --campaign DIR. Finish should use DIR/sources/experiments/hybrid_selection_round.py.
A fresh selector receives only selector/ball/trial/round/context.json and instructions.txt.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'apps/experiment-design'))
from packages.dynamics.ball_identification import Observations, predict, fit_classical, validate_observations
from packages.dynamics.bouncing_ball import BouncingBall
from packages.pinn.ball import BallPINN, train, tensor
from selector import SYSTEM_INSTRUCTIONS, parse_decision, validate_context
import round_store

TIMES = [.08,.14,.20,.26,.32,.38,.44,.50,.56,.62,.68,.70]
CANDIDATES = [('downward',-4.,101),('neutral',0.,202),('upward',4.,303)]
SOURCE_NAMES = ['experiments/hybrid_selection_round.py','packages/pinn/ball.py',
                'packages/dynamics/ball_identification.py','packages/dynamics/bouncing_ball.py',
                'packages/dynamics/__init__.py','packages/pinn/__init__.py',
                'apps/experiment-design/selector.py','apps/experiment-design/round_store.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    with Path(path).open('x') as stream:
        json.dump(obj,stream,indent=2,allow_nan=False); stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def trace(root,kind,summary,paths):
    with (root/'thoughts.jsonl').open('a') as stream:
        stream.write(json.dumps({'timestamp_utc':now(),'kind':kind,'summary':summary,
                    'artifacts_sha256':{str(p.relative_to(root)):sha(p) for p in paths},
                    'usage':{'source':'collaboration API does not expose token counts or model identity',
                             'model_id':None,'input_tokens':None,'output_tokens':None,
                             'cached_input_tokens':None,'reasoning_tokens':None}})+'\n')


def serialize(obs):
    return {**asdict(obs),'times':obs.times.tolist(),'heights':obs.heights.tolist()}


def deserialize(value):
    return Observations(value['initial_height'],value['acceleration'],np.array(value['times']),np.array(value['heights']))


def generate(protocol,acceleration,seed):
    truth = protocol['simulator']
    t = np.array(protocol.get('data',{}).get('times_s',TIMES))
    y,_ = predict(truth['floor_m'],truth['restitution'],1.,acceleration,t)
    return Observations(1.,acceleration,t,y+np.random.default_rng(seed).normal(0,.01,len(t)))


def source_check(root):
    for name,digest in read(root/'protocol.json')['source_sha256'].items():
        if sha(root/'sources'/name)!=digest:
            raise ValueError(f'frozen source modified: {name}')


def warm_start(baseline,observations,protocol):
    model = BallPINN(observations,protocol['inference'],protocol['pinn'],seed=0)
    old = baseline['model_state_dict']
    state = model.state_dict()
    for key,value in old.items():
        if key in ('raw_times','tau_bounds'):
            continue
        state[key] = value.clone()
    state['raw_times'][0]=old['raw_times'][0]
    model.load_state_dict(state)
    return model


def state_digest(state,keys):
    result = hashlib.sha256()
    for key in sorted(keys):
        result.update(key.encode()); result.update(state[key].detach().cpu().numpy().tobytes())
    return result.hexdigest()


def information(floor,restitution,acceleration,inference,times=None):
    times=np.asarray(TIMES if times is None else times)
    lower = np.array([inference['floor_bounds_m'][0],inference['restitution_bounds'][0]])
    upper = np.array([inference['floor_bounds_m'][1],inference['restitution_bounds'][1]])
    ranges = upper-lower
    q = (np.array([floor,restitution])-lower)/ranges
    columns=[]
    for axis in range(2):
        left=q.copy(); right=q.copy()
        left[axis]=max(0.,q[axis]-1e-5); right[axis]=min(1.,q[axis]+1e-5)
        lp=lower+left*ranges; rp=lower+right*ranges
        a=predict(*lp,1.,acceleration,times)[0]
        b=predict(*rp,1.,acceleration,times)[0]
        columns.append((b-a)/(right[axis]-left[axis])/.01)
    jac=np.stack(columns,axis=1)
    return jac.T@jac


def selector_context(baseline,protocol):
    params=baseline['parameters']; inference=protocol['inference']
    times=list(protocol.get('data',{}).get('times_s',TIMES))
    initial=information(params['floor_m'],params['restitution'],0.,inference,times)
    eigenvalues=np.linalg.eigvalsh(initial)
    rank=int(np.sum(eigenvalues>1e-10*max(float(eigenvalues[-1]),1.)))
    candidates=[]
    for identifier,acceleration,_ in CANDIDATES:
        validate_observations([Observations(1.,acceleration,np.array(times),np.zeros(len(times)))],
                              inference['floor_bounds_m'],inference['restitution_bounds'],9.81)
        candidate=information(params['floor_m'],params['restitution'],acceleration,inference,times)
        score=float(np.linalg.slogdet(initial+candidate+1e-9*np.eye(2))[1])
        candidates.append({'id':identifier,'acceleration_m_s2':acceleration,'initial_height_m':1.,
                           'times_s':times,'observations':len(times),'predicted_information_logdet':score,
                           'predicted_information_eigenvalues':np.linalg.eigvalsh(initial+candidate).tolist()})
    return {'objective':'Choose one 12-observation actuation to improve shared floor/restitution identification after a neutral trajectory.',
            'history':[{'candidate_id':'neutral','acceleration_m_s2':0.,'initial_height_m':1.,
                        'times_s':times,'observations':len(times)}],
            'diagnostics':{'learned_parameters':params,'estimated_event_times_s':baseline['learned_event_times_s'],
                           'pinn':baseline['diagnostics'],'noise_std_m':.01,'inference_prior':inference,
                           'information_rule':'Position finite-difference Jacobian in dimensionless prior coordinates, scaled by noise; logdet(F_initial+F_candidate+1e-9 I). Local prediction, not measured error.',
                           'numerical_policy':'Maximize predicted_information_logdet; ties choose first listed candidate.',
                           'initial_information_eigenvalues':eigenvalues.tolist(),
                           'initial_information_rank':rank,
                           'initial_information_condition_number':float(eigenvalues[-1]/eigenvalues[0]) if rank==2 else None,
                           'information_rank_definition':'Unregularized initial Fisher eigenvalues above1e-10 times max(largest eigenvalue,1); condition unavailable when rank deficient.'},
            'candidates':candidates}


def parameter_transfer_metrics(parameters,protocol):
    truth=protocol['simulator'];floor=parameters['floor_m'];restitution=parameters['restitution']
    errors=[];timings=[]
    for h0 in [.85,1.15]:
        for acceleration in [-2.,2.]:
            times=np.linspace(0,.62,301)
            true,_=predict(truth['floor_m'],truth['restitution'],h0,acceleration,times)
            fitted,_=predict(floor,restitution,h0,acceleration,times)
            errors.extend(fitted-true)
            timings.append(abs(np.sqrt(2*(h0-floor)/(9.81-acceleration))-np.sqrt(2*(h0-truth['floor_m'])/(9.81-acceleration))))
    return {'primary_metric_definition':'Analytic parameter-transfer position RMSE on fixed reserved inputs; NOT direct neural generalization.',
            'parameter_transfer_position_rmse_m':float(np.sqrt(np.mean(np.square(errors)))),
            'parameter_transfer_impact_time_mae_s':float(np.mean(timings)),
            'floor_absolute_error_m':abs(floor-truth['floor_m']),
            'restitution_absolute_error':abs(restitution-truth['restitution'])}


def evaluate(model,observations,protocol):
    truth=protocol['simulator']; floor=float(model.floor.detach()); restitution=float(model.restitution.detach())
    metrics=parameter_transfer_metrics({'floor_m':floor,'restitution':restitution},protocol)
    reconstruction=[]; observed_timing=[]; resets=[]
    for index,obs in enumerate(observations):
        times=np.linspace(0,float(obs.times[-1]),301)
        true,_=predict(truth['floor_m'],truth['restitution'],obs.initial_height,obs.acceleration,times)
        with torch.no_grad():
            fitted,_=model(index,times)
            reconstruction.extend(fitted.numpy().ravel()-true)
            _,vpre=model.segment(index,tensor([[1.]]))
            _,vpost=model.segment(index,tensor([[0.]]),post=True)
            resets.append(abs(float(vpost+model.restitution*vpre)))
        observed_timing.append(abs(float(model.event_times[index].detach())-np.sqrt(2*(obs.initial_height-truth['floor_m'])/(9.81-obs.acceleration))))
    return {**metrics,
            'direct_neural_reconstruction_position_rmse_m':float(np.sqrt(np.mean(np.square(reconstruction)))),
            'observed_trajectory_event_time_mae_s':float(np.mean(observed_timing)),
            'reset_velocity_absolute_error_m_s':max(resets)}


def worker(root,job_name):
    source_check(root)
    protocol=read(root/'protocol.json'); job=root/job_name
    observations=[deserialize(o) for o in read(job/'observations.json')]
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if job_name=='baseline':
        model=BallPINN(observations,protocol['inference'],protocol['pinn'],seed=0)
    else:
        baseline=torch.load(root/'baseline/checkpoint.pt',weights_only=True)
        model=warm_start(baseline,observations,protocol)
        state=model.state_dict(); old=baseline['model_state_dict']
        old_keys=[key for key in old if key not in ['raw_times','tau_bounds']]
        if state_digest(state,old_keys)!=state_digest(old,old_keys) or not torch.equal(state['raw_times'][:1],old['raw_times']):
            raise ValueError('baseline warm-start mismatch')
        write(job/'initial-state.json',{'baseline_checkpoint_sha256':sha(root/'baseline/checkpoint.pt'),
              'copied_baseline_state_sha256':state_digest(state,old_keys),
              'existing_raw_event_time':state['raw_times'][0].item(),
              'new_modules_sha256':state_digest(state,[k for k in state if k.startswith(('pre.1.','post.1.'))]),
              'new_raw_event_time':state['raw_times'][1].item()})
        torch.save({'model_state_dict':state},job/'initial-checkpoint.pt')
    try:
        with (job/'training.jsonl').open('x') as log:
            def progress(row):
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row),flush=True)
            training=train(model,progress)
        torch.save({'model_state_dict':model.state_dict()},job/'checkpoint.pt')
        parameters={'floor_m':float(model.floor.detach()),'restitution':float(model.restitution.detach())}
        classical=fit_classical(observations,protocol['inference']['floor_bounds_m'],protocol['inference']['restitution_bounds'])
        classical['evaluation']=parameter_transfer_metrics(classical,protocol)
        result={'status':'complete','parameters':parameters,'training':training,
                'learned_event_times_s':model.event_times.detach().tolist(),
                'diagnostics':model.diagnostics(197),'evaluation':evaluate(model,observations,protocol),
                'classical':classical,'completed_at_utc':now()}
        write(job/'result.json',result)
    except Exception:
        torch.save({'model_state_dict':model.state_dict()},job/'failed-checkpoint.pt')
        (job/'worker-stack.txt').write_text(traceback.format_exc())
        raise


def run_job(root,name):
    launches=list(root.glob('*/invocation.json'))
    budgets=read(root/'protocol.json')['budgets']
    if len(launches)>=budgets['maximum_training_jobs']:
        raise ValueError('maximum training jobs already launched')
    elapsed=sum(read(p)['elapsed_s'] for p in root.glob('*/process.json'))
    remaining=budgets['total_compute_s']-elapsed-budgets.get('cleanup_reserve_s',0.)
    if remaining<=0:
        status={'status':'training_budget_unavailable','elapsed_s':0.,'completed_at_utc':now()}
        write(root/name/'process.json',status)
        return status
    job=root/name
    command=[sys.executable,str(root/'sources/experiments/hybrid_selection_round.py'),'worker','--campaign',str(root),'--job',name]
    write(job/'invocation.json',{'argv':command,'started_at_utc':now(),'timeout_s':min(budgets['job_timeout_s'],remaining)})
    return subprocess_job(job,command,min(budgets['job_timeout_s'],remaining))


def subprocess_job(job,command,timeout):
    start=time.monotonic()
    with (job/'stdout.log').open('x') as out,(job/'stderr.log').open('x') as err:
        try:
            result=subprocess.run(command,stdout=out,stderr=err,timeout=timeout,check=True)
            status={'status':'worker_finished','returncode':result.returncode}
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
            (job/'controller-stack.txt').write_text(traceback.format_exc())
            status={'status':'worker_failed','error_type':type(exc).__name__,'error':str(exc)}
    status.update({'elapsed_s':time.monotonic()-start,'completed_at_utc':now()})
    write(job/'process.json',status)
    return status


def inventory(root,exclude):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in exclude}


def check_manifest(root,name):
    manifest=read(root/name)
    for relative,digest in manifest['artifacts_sha256'].items():
        path=root/relative
        if not path.resolve().is_relative_to(root.resolve()) or sha(path)!=digest:
            raise ValueError(f'artifact hash mismatch: {relative}')
    return manifest


def prepare(root, overrides=None, defer_training=False):
    root.mkdir(parents=True,exist_ok=False)
    overrides=overrides or {}
    names=SOURCE_NAMES+overrides.get('extra_source_names',[])
    sources=root/'sources'
    for name in names:
        target=sources/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target)
    protocol={'study':'one synthetic development hybrid selection round','frozen_at_utc':now(),
              'simulator':{'floor_m':.12,'restitution':.8},
              'inference':{'floor_bounds_m':[0.,.25],'restitution_bounds':[.6,.95],'initial_floor_m':.05,'initial_restitution':.65},
              'pinn':{'width':16,'collocation_points_per_segment':48,'adam_steps':1500,'adam_learning_rate':.003,
                      'lbfgs_steps':1000,'physics_weight':100.,'guard_weight':100.},
              'data':{'initial_height_m':1.,'times_s':TIMES,'noise_std_m':.01,'baseline_seed':17,
                      'candidate_seed_offsets':{identifier:offset for identifier,_,offset in CANDIDATES}},
              'evaluation':{'initial_heights_m':[.85,1.15],'accelerations_m_s2':[-2.,2.],
                            'horizon_s':.62,'points_per_trajectory':301,'definition':'Analytic parameter transfer, not direct NN generalization'},
              'budgets':{'maximum_training_jobs':3,'job_timeout_s':120,'total_compute_s':600},
              'information_selector':{'coordinates':'parameters normalized by prior ranges','difference_step':1e-5,'noise_std':.01,'jitter':1e-9,'tie_break':'listed candidate order'},
              'source_sha256':{name:sha(sources/name) for name in names}}
    for section in ['data','budgets','simulator']:
        protocol[section].update(overrides.get(section,{}))
    write(root/'protocol.json',protocol)
    write(root/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                                  'scipy':scipy.__version__,'device':'cpu','dtype':'float64','threads':1,'invocation':sys.argv})
    baseline=root/'baseline'; baseline.mkdir()
    write(baseline/'observations.json',[serialize(generate(protocol,0.,protocol['data']['baseline_seed']))])
    trace(root,'hypothesis','One new actuation may improve shared contact-parameter recovery; compare fresh-agent choice to frozen information rule from identical baseline.',[root/'protocol.json',root/'environment.json',baseline/'observations.json'])
    if defer_training:
        return
    complete_prepare(root)


def complete_prepare(root):
    source_check(root)
    protocol=read(root/'protocol.json');baseline=root/'baseline'
    status=run_job(root,'baseline')
    trace(root,'evidence','Baseline job retained with bounded subprocess logs and diagnostics.',list(baseline.iterdir()))
    if status['status']!='worker_finished':
        write(root/'manifest.json',{'status':'baseline_failed','artifacts_sha256':inventory(root,{'manifest.json'})})
        return
    context=selector_context(read(baseline/'result.json'),protocol)
    validate_context(context)
    write(root/'context-before-choice.json',context)
    (root/'selector-instructions.txt').write_text(SYSTEM_INSTRUCTIONS)
    target=round_store.start(argparse.Namespace(root=root/'selector',study='ball',trial='trial',round='round',
            selector='fresh-collaboration-agent',model_id=None,prompt_version='select-v1',
            context=root/'context-before-choice.json',instructions=root/'selector-instructions.txt'))
    numerical=max(context['candidates'],key=lambda c:c['predicted_information_logdet'])
    write(root/'numerical-decision.json',{'candidate_id':numerical['id'],'hypothesis':'Higher local information may improve parameter identifiability.',
                                        'rationale':'Frozen maximum logdet information rule, listed-order tie break.'})
    if 'random_policy_seed' in protocol['data']:
        identifier=uniform_random_candidate(protocol['data']['random_policy_seed'])
        write(root/'random-decision.json',{'candidate_id':identifier,'hypothesis':'Uniform random actuation is an untuned policy comparator.',
                                          'rationale':'Independent frozen seeded uniform draw from the listed candidates.'})
    trace(root,'decision','Freeze selector-visible evidence and exact instructions before any fresh-agent selection; reserved scores and simulator truth are excluded.',[target/'context.json',target/'instructions.txt',root/'numerical-decision.json']+([root/'random-decision.json'] if (root/'random-decision.json').exists() else []))
    mutable={'thoughts.jsonl','selector/ball/trial/round/manifest.json','prepare-manifest.json'}
    write(root/'prepare-manifest.json',{'status':'awaiting_selector','artifacts_sha256':inventory(root,mutable)})
    print(f'SELECTOR_CONTEXT={target/"context.json"}\nSELECTOR_INSTRUCTIONS={target/"instructions.txt"}',flush=True)


def uniform_random_candidate(seed):
    return CANDIDATES[int(np.random.default_rng(seed).integers(0,len(CANDIDATES)))][0]


def finish(root,decision_path,metadata_path):
    source_check(root); verify_prepared(root)
    target=root/'selector/ball/trial/round'
    context=read(target/'context.json')
    decision=parse_decision(decision_path.read_text(),validate_context(context))
    # Exclusive marker prevents repeating or expanding this campaign after failure.
    write(root/'finish-lock.json',{'started_at_utc':now()})
    shutil.copyfile(decision_path,root/'selector-response.json')
    shutil.copyfile(metadata_path,root/'selector-task-metadata.json')
    write(root/'usage.json',{'source':'collaboration API does not expose token usage','elapsed_ms':None,
                           'input_tokens':None,'cached_input_tokens':None,'output_tokens':None,'reasoning_tokens':None})
    trace(root,'decision','Fresh selector response is validated without modification; numerical choice remains frozen.',[root/'selector-response.json',root/'selector-task-metadata.json',root/'numerical-decision.json'])
    protocol=read(root/'protocol.json')
    decisions={'llm':decision.candidate_id,'numerical':read(root/'numerical-decision.json')['candidate_id']}
    if (root/'random-decision.json').exists():
        decisions['random']=read(root/'random-decision.json')['candidate_id']
    for policy,candidate in decisions.items():
        acceleration,offset=next((a,o) for i,a,o in CANDIDATES if i==candidate)
        job=root/policy; job.mkdir()
        old=read(root/'baseline/observations.json')
        new=serialize(generate(protocol,acceleration,protocol['data'].get('candidate_noise_seed_base',17)+protocol['data']['candidate_seed_offsets'][candidate]))
        write(job/'new-observations.json',new)
        write(job/'observations.json',old+[new])
        trace(root,'action',f'{policy} selected {candidate}; new evaluator stream is keyed by candidate and shared between policies for identical choices.',[job/'new-observations.json',job/'observations.json',root/'baseline/checkpoint.pt'])
        run_job(root,policy)
        trace(root,'evidence',f'{policy} update outputs retained from baseline warm start under equal optimizer budgets.',list(job.iterdir()))
    policies={}
    for policy,candidate in decisions.items():
        result_path=root/policy/'result.json'
        policies[policy]={'candidate_id':candidate,'result':read(result_path) if result_path.exists() else read(root/policy/'process.json')}
    result={'candidate_id':decision.candidate_id,'policies':policies,'completed_at_utc':now()}
    write(root/'round-result.json',result)
    round_store.finish(argparse.Namespace(directory=target,decision=root/'selector-response.json',result=root/'round-result.json',usage=root/'usage.json'))
    lines=['# One synthetic hybrid selection round','',
           'Primary metric is analytic parameter-transfer position RMSE on reserved inputs, not direct neural generalization. One round and one noise realization cannot establish selector superiority.', '',
           '|Fit|Candidate|Status|Transfer RMSE|Floor error|Restitution error|Timing MAE|Neural reconstruction RMSE|',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for name in ['baseline',*decisions]:
        path=root/name/'result.json'
        if not path.exists():
            lines.append(f'|{name}|{decisions.get(name,"neutral")}|failed|—|—|—|—|—|'); continue
        r=read(path); e=r['evaluation']
        lines.append(f"|{name}|{decisions.get(name,'neutral')}|{r['status']}|{e['parameter_transfer_position_rmse_m']:.6f}|{e['floor_absolute_error_m']:.6f}|{e['restitution_absolute_error']:.6f}|{e['parameter_transfer_impact_time_mae_s']:.6f}|{e['direct_neural_reconstruction_position_rmse_m']:.6f}|")
    lines+=['', 'Both policies selected the same candidate; this validates pipeline repeatability, not a policy advantage.' if decisions['llm']==decisions['numerical'] else 'The policies chose different candidates; this single-round difference is descriptive, not evidence of policy superiority.', '', 'Flow equations: dh/dt=v, dv/dt=u−g. Contact guard h=floor and velocity reset v+=−restitution*v− are handled by piecewise neural segments and latent event times. Guard/flow diagnostics are measured separately; reset and initial conditions are enforced by construction. Detailed diagnostics, classical comparator and reset errors are in each result.json.', '',
            'All updated fits copy the same baseline weights, learned physical parameters and old event-time parameter. Additional trajectory modules use the same seed. Candidate-keyed observations are identical whenever choices match; each policy still receives its own training job.', '',
            'Learned physical parameters are listed independently of checkpoint neural module weights. Fixed full budgets: 1500 Adam steps, at most 1000 L-BFGS iterations; actual closure calls are reported. Failures preserve progress/logs/stacks and are not silently retried. Token/model usage unavailable from collaboration is null.', '',
            'This benchmark requires the entire prior to permit exactly one impact before the horizon and excludes the second impact. It does not test Zeno continuation or multiple impacts.', '',
            'Next step: inspect transfer error changes alongside flow/guard diagnostics; agreement or disagreement with a local information proxy in one round is descriptive only.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    trace(root,'next_step','Review baseline-to-update transfer errors with hybrid residuals before designing a larger independent replication.',[root/'round-result.json',root/'report.md',target/'decision.json',target/'result.json'])
    completed=all((root/p/'result.json').exists() for p in ['baseline',*decisions])
    write(root/'manifest.json',{'status':'complete' if completed else 'completed_with_failures',
          'compute_elapsed_s':sum(read(p)['elapsed_s'] for p in root.glob('*/process.json')),
          'artifacts_sha256':inventory(root,{'manifest.json'})})
    verify(root)
    print(f'Completed and verified {root}',flush=True)


def verify_prepared(root):
    manifest=check_manifest(root,'prepare-manifest.json')
    target=root/'selector/ball/trial/round'
    actual={p.name for p in target.iterdir() if p.is_file()}
    if actual != {'context.json','instructions.txt','manifest.json'}:
        raise ValueError('prepared selector inventory mismatch')
    selector_manifest=read(target/'manifest.json')
    if selector_manifest['status']!='open':
        raise ValueError('selector round is not open')
    for name,digest in selector_manifest['files_sha256'].items():
        if sha(target/name)!=digest:
            raise ValueError('prepared selector hash mismatch')
    return manifest


def verify(root):
    source_check(root)
    if not (root/'manifest.json').exists():
        verify_prepared(root); print('Prepared immutable artifacts verified'); return
    manifest=check_manifest(root,'manifest.json')
    if set(manifest['artifacts_sha256'])!=set(inventory(root,{'manifest.json'})):
        raise ValueError('artifact inventory mismatch')
    for row in map(json.loads,(root/'thoughts.jsonl').read_text().splitlines()):
        for relative,digest in row['artifacts_sha256'].items():
            if manifest['artifacts_sha256'].get(relative)!=digest:
                raise ValueError('trace artifact mismatch')
    print('Campaign artifacts and visible trace verified')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','finish','verify','worker'])
    parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path)
    parser.add_argument('--decision',type=Path);parser.add_argument('--metadata',type=Path);parser.add_argument('--job')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out.resolve())
    elif args.command=='finish':finish(args.campaign.resolve(),args.decision,args.metadata)
    elif args.command=='verify':verify(args.campaign.resolve())
    else:worker(args.campaign.resolve(),args.job)


if __name__=='__main__':
    main()
