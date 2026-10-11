"""Frozen candidate-menu expansion pilot:3vs9 inputs, two sequential acquisitions.

prepare --out NEW; acquire --campaign DIR --round1or2 --decision-map JSON;
verify --campaign DIR. Use frozen DIR/sources/experiments/hybrid_scale_pilot.py.
Four worlds; five policies;40 final PINNs +4 shared initial fits,600s maximum.
LLM contexts never include design utilities, rankings, truth or reserved scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import os

for _thread_key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_thread_key]='1'

import numpy as np
import scipy
from scipy.special import logsumexp
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import hybrid_selection_round as core
import hybrid_physical_sweep as physical

SELF='experiments/hybrid_scale_pilot.py'
POLICIES=['fresh_llm','posterior_numerical','fixed_downward','fixed_upward','uniform_random']
MENUS={3:[-4,0,4],9:list(range(-4,5))}
PRIMARY='parameter_transfer_position_rmse_m'
INSTRUCTIONS='''Choose one next acceleration experiment to identify shared floor and restitution.
Use only supplied observed history, known model/prior, uncertainty diagnostics and unranked candidates.
There are two sequential choices in total; the next observation is not available yet.
Return ONLY JSON keys candidate_id, hypothesis, rationale. Choose an exact supplied ID.
Each text field must be nonempty and under500characters. Do not change model, times, loss or training.
'''


def candidate_id(u):return f'acceleration-{u:+d}'


def design():
    rows=[]
    for floor in [.045,.205]:
        for restitution in [.70,.86]:
            index=len(rows)
            rows.append({'case_id':f'case-{index+1}','floor_m':floor,'restitution':restitution,
                         'baseline_noise_seed':1400001+index,
                         'candidate_noise_seeds':{str(r):{candidate_id(u):1600000+10000*index+1000*r+i for i,u in enumerate(MENUS[9])} for r in [1,2]},
                         'random_policy_seeds':{str(r):1800000+100*index+r for r in [1,2]},
                         'design_mc_seeds':{str(r):2000000+100*index+r for r in [1,2]}})
    return rows


def posterior(observations,inference,resolution=129):
    started=time.monotonic()
    floor,e=np.meshgrid(np.linspace(*inference['floor_bounds_m'],resolution),np.linspace(*inference['restitution_bounds'],resolution),indexing='ij')
    parameters=np.column_stack([floor.ravel(),e.ravel()]);logweights=np.zeros(len(parameters))
    for value in observations:
        obs=core.deserialize(value)
        if obs.initial_height!=1.:raise ValueError('posterior supports only declared releaseheight1')
        means=predictions(parameters,obs.acceleration,obs.times)
        logweights-=np.sum((means-obs.heights[None,:])**2,axis=1)/(2*.01**2)
    logweights-=logsumexp(logweights);weights=np.exp(logweights)
    order=np.argsort(-weights,kind='stable');count=int(np.searchsorted(np.cumsum(weights[order]),1-1e-12)+1)
    keep=np.sort(order[:min(count,len(order))]);retained=float(weights[keep].sum())
    activeweights=weights[keep]/retained
    mean=np.sum(parameters*weights[:,None],axis=0);sd=np.sqrt(np.sum((parameters-mean)**2*weights[:,None],axis=0))
    summary={}
    for axis,name in enumerate(['floor_m','restitution']):
        indices=np.argsort(parameters[:,axis],kind='stable');cdf=np.cumsum(weights[indices])
        summary[name]={'mean':float(mean[axis]),'std':float(sd[axis]),
                       'quantiles_05_50_95':[float(parameters[indices[min(np.searchsorted(cdf,p),len(indices)-1)],axis]) for p in [.05,.5,.95]],
                       'grid_spacing':float((parameters[:,axis].max()-parameters[:,axis].min())/(resolution-1)),
                       'spacing_to_std_ratio':float((parameters[:,axis].max()-parameters[:,axis].min())/(resolution-1)/max(sd[axis],1e-15))}
    boundary=(floor.ravel()==floor.min())|(floor.ravel()==floor.max())|(e.ravel()==e.min())|(e.ravel()==e.max())
    diagnostic={'resolution':resolution,'parameter_summary':summary,'effective_grid_count':float(1/np.sum(weights**2)),
                'boundary_mass':float(weights[boundary].sum()),'active_grid_count':len(keep),'retained_mass':retained,
                'interpretation':'Uniform bounded discrete prior and exact Gaussian observation likelihood; uncertainty is grid-based, model/noise assumptions may be wrong.'}
    return {'parameters':parameters[keep],'weights':activeweights,'summary':diagnostic,'elapsed_s':time.monotonic()-started}


def predictions(parameters,u,times):
    p=np.atleast_2d(parameters);t=np.asarray(times)[None,:];a=9.81-u;tau=np.sqrt(2*(1-p[:,0,None])/a);dt=t-tau
    return np.where(t<tau,1-.5*a*t*t,p[:,0,None]+p[:,1,None]*a*tau*dt-.5*a*dt*dt)


def information_scores(post,times,actions,seed,draws=128):
    started=time.monotonic();rng=np.random.default_rng(seed)
    uniforms=rng.random(draws);indices=np.searchsorted(np.cumsum(post['weights']),uniforms,side='right')
    indices=np.minimum(indices,len(post['weights'])-1);noise=rng.normal(size=(draws,len(times)))
    scores={};contributions={}
    for u in actions:
        means=predictions(post['parameters'],u,times);y=means[indices]+.01*noise
        squared=np.maximum(np.sum(y*y,axis=1)[:,None]+np.sum(means*means,axis=1)[None,:]-2*y@means.T,0.)
        conditional=-.5*np.sum(noise*noise,axis=1)
        mixture=logsumexp(np.log(post['weights'])[None,:]-.5*squared/.01**2,axis=1)
        values=conditional-mixture
        identifier=candidate_id(u);scores[identifier]={'mean':float(values.mean()),'mc_standard_error':float(values.std(ddof=1)/np.sqrt(draws))}
        contributions[identifier]=values
    ordered=sorted(actions,key=lambda u:-scores[candidate_id(u)]['mean'])
    chosen=candidate_id(ordered[0]);second=candidate_id(ordered[1]);difference=contributions[chosen]-contributions[second]
    gap={'runner_up':second,'mean_difference':float(difference.mean()),'paired_mc_standard_error':float(difference.std(ddof=1)/np.sqrt(draws))}
    return {'selected_candidate_id':chosen,'scores':scores,'selected_gap':gap,'draws':draws,'seed':seed,
            'elapsed_s':time.monotonic()-started,'rule':'Predictive mutual information MC mean, stable Gaussian mixture, same latent/noise draws across candidates; negative estimates retained.',
            'contributions':{k:v.tolist() for k,v in contributions.items()}}


def source_check(root):
    for name,digest in core.read(root/'protocol.json')['source_sha256'].items():
        if core.sha(root/'sources'/name)!=digest:raise ValueError(f'frozen source mismatch:{name}')


def processes(case):return list(case.glob('baseline/process.json'))+list(case.glob('routes/*/*/process.json'))


def launch(case,job,command):
    budget=core.read(case/'protocol.json')['budgets'];existing=list(case.glob('baseline/invocation.json'))+list(case.glob('routes/*/*/invocation.json'))
    if len(existing)>=11:raise ValueError('case training-job cap exceeded')
    elapsed=sum(core.read(p)['elapsed_s'] for p in processes(case));remaining=150-elapsed-.5
    if remaining<=0:
        result={'status':'training_budget_unavailable','elapsed_s':0.,'completed_at_utc':core.now()};core.write(job/'process.json',result);return result
    timeout=min(120,remaining)
    core.write(job/'invocation.json',{'argv':command,'timeout_s':timeout,'started_at_utc':core.now()})
    return core.subprocess_job(job,command,timeout)


def fit_classical(observations,protocol):
    obs=[core.deserialize(o) for o in observations]
    result=core.fit_classical(obs,protocol['inference']['floor_bounds_m'],protocol['inference']['restitution_bounds'])
    result['evaluation']=core.parameter_transfer_metrics(result,protocol)
    return result


def route_path(case,size,policy):return case/'routes'/str(size)/policy


def context_for(case,route,size,round_number,history,post,classical):
    protocol=core.read(case/'protocol.json');initial=core.read(case/'baseline/result.json')
    return {'objective':'Choose the next actuation to identify floor and restitution over two sequential experiments.',
            'round':round_number,'rounds_total':2,'history':[{'initial_height_m':o['initial_height'],'acceleration_m_s2':o['acceleration'],
                         'times_s':o['times'],'observed_heights_m':o['heights']} for o in history],
            'known_model':'Release at height1m with zero velocity. Flow dh/dt=v,dv/dt=u−9.81. Impact at h=floor resets v+=−restitution*v−. One rebound is supported by the entire prior.',
            'diagnostics':{'noise_std_m':.01,'prior':{'floor_bounds_m':protocol['inference']['floor_bounds_m'],'restitution_bounds':protocol['inference']['restitution_bounds']},
                'current_classical_fit':{k:classical[k] for k in ['floor_m','restitution','observation_rmse_m','optimizer_success','jacobian_singular_values']},
                'posterior':post['summary'],'initial_pinn_diagnostics':initial['diagnostics'],
                'initial_pinn_status':'Initial12-observation fit only; stale after round1. No intermediate neural retraining.',
                'limitations':'Grid posterior and classical estimates depend on assumed dynamics/Gaussian noise; no calibrated real-world posterior guarantee.'},
            'candidates':[{'id':candidate_id(u),'acceleration_m_s2':u,'initial_height_m':1.,'times_s':protocol['data']['times_s'],'observations':12,
                           'description':f'Known constant input {u:+d}m/s²; effective downward acceleration {9.81-u:.2f}m/s². Same observation window/cost.'} for u in MENUS[size]]}


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);names=core.SOURCE_NAMES+['experiments/hybrid_physical_sweep.py',SELF];sources=root/'sources'
    for name in names:
        target=sources/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    common=physical.common();common['data']['times_s']=np.linspace(.56,.7,12).tolist();common['budgets'].update({'maximum_training_jobs':11,'total_compute_s':150})
    for u in MENUS[9]:
        core.validate_observations([core.Observations(1.,u,np.array(common['data']['times_s']),np.zeros(12))],common['inference']['floor_bounds_m'],common['inference']['restitution_bounds'],9.81)
    protocol={'study':'Candidate-menu expansion/two-step planning pilot, not model scaling or confirmation','frozen_at_utc':core.now(),
              'cases':design(),'common':common,'menus':MENUS,'policies':POLICIES,'rounds':2,'final_observation_count':36,'model_seed':0,
              'maximum_training_jobs':44,'maximum_pinn_compute_s':600,'per_case_pinn_compute_s':150,'job_timeout_s':120,'budget_borrowing':False,
              'source_sha256':{name:core.sha(sources/name) for name in names},
              'numerical_rule':{'primary_grid':129,'diagnostic_grid':257,'mc_draws':128,'active_retained_mass':1-1e-12,
                    'selection':'Greedy sequential expected predictive mutual information; choose largest MCmean, listed-order ties. Diagnostic257 choice never replaces primary.',
                    'uncertainty':'MCSE and paired gapSE, grid spacing/std, ESS/boundary/retainedmass; no adaptive draws/resolution tuning.'},
              'analysis':'Four matched physical cases, no noise replicates. Compare3vs9 changes and paired policy-minus-comparator interaction; descriptive only. Any incomplete route suppresses aggregate ranking.',
              'compute_fairness':'Equal measurements and final training budgets, NOT equal total compute; record numerical/classical/policy/selector wall separately.'}
    core.write(root/'protocol.json',protocol)
    core.write(root/'environment.json',{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
               'device':'cpu','dtype':'float64','blas_threads':1,'torch_training_threads':1,'invocation':sys.argv,'started_at_utc':core.now()})
    cases=root/'cases';cases.mkdir()
    for spec in protocol['cases']:
        case=cases/spec['case_id'];case.mkdir();p=physical.common();p['data']['times_s']=common['data']['times_s'];p['budgets']=common['budgets']
        p['simulator'].update({'floor_m':spec['floor_m'],'restitution':spec['restitution']});p.update({'case_spec':spec,'source_sha256':protocol['source_sha256']})
        for name in names:
            target=case/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(sources/name,target)
        core.write(case/'protocol.json',p);baseline=case/'baseline';baseline.mkdir()
        core.write(baseline/'observations.json',[core.serialize(core.generate(p,0.,spec['baseline_noise_seed']))])
        core.trace(case,'hypothesis','Matched menus share baseline data, two equal-cost acquisition rounds and candidate-keyed evaluator noise.',[case/'protocol.json',baseline/'observations.json'])
    excluded={'input-manifest.json','thoughts.jsonl'}|{f"cases/{s['case_id']}/thoughts.jsonl" for s in protocol['cases']}
    core.write(root/'input-manifest.json',{'artifacts_sha256':core.inventory(root,excluded)})
    core.trace(root,'hypothesis','Test candidate-menu expansion and two sequential selections with actual unscored-context LLM choices against uncertainty-aware numerical and both fixed directions.',[root/'protocol.json',root/'input-manifest.json'])
    command=[sys.executable,str(sources/SELF),'prepare-initial','--campaign',str(root)]
    try:subprocess.run(command,check=True,env=thread_env())
    except Exception:(root/'prepare-stack.txt').write_text(traceback.format_exc());raise


def thread_env():
    import os
    return {**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}


def numerical_decision(root,case,route,size,round_number,history):
    p=core.read(case/'protocol.json');seed=p['case_spec']['design_mc_seeds'][str(round_number)]
    post=posterior(history,p['inference'],129);refined=posterior(history,p['inference'],257)
    for label,value in [('primary',post),('refinement',refined)]:
        target=route/f'round-{round_number}-{label}-posterior.npz'
        with target.open('xb') as stream:np.savez_compressed(stream,parameters=value['parameters'],weights=value['weights'])
    primary=information_scores(post,p['data']['times_s'],MENUS[size],seed);check=information_scores(refined,p['data']['times_s'],MENUS[size],seed)
    record={'primary':primary,'refinement':check,'primary_posterior':post['summary'],'refined_posterior':refined['summary'],
            'choice_stable':primary['selected_candidate_id']==check['selected_candidate_id'],
            'compute_elapsed_s':post['elapsed_s']+refined['elapsed_s']+primary['elapsed_s']+check['elapsed_s']}
    core.write(route/f'round-{round_number}-numerical.json',record)
    return primary['selected_candidate_id']


def freeze_round(root,round_number):
    contexts=[]
    for spec in core.read(root/'protocol.json')['cases']:
        case=root/'cases'/spec['case_id'];p=core.read(case/'protocol.json')
        for size in MENUS:
            for policy in POLICIES:
                route=route_path(case,size,policy)
                if round_number==1:route.mkdir(parents=True)
                history=core.read(case/'baseline/observations.json') if round_number==1 else core.read(route/'round-1-observations.json')
                classical_started=time.monotonic();classical=fit_classical(history,p);classical_elapsed=time.monotonic()-classical_started
                core.write(route/f'round-{round_number}-before-classical.json',{'fit':classical,'elapsed_s':classical_elapsed})
                if policy=='posterior_numerical':choice=numerical_decision(root,case,route,size,round_number,history)
                elif policy=='fixed_downward':choice=candidate_id(-4)
                elif policy=='fixed_upward':choice=candidate_id(4)
                elif policy=='uniform_random':
                    u=np.random.default_rng(spec['random_policy_seeds'][str(round_number)]).random();choice=candidate_id(MENUS[size][min(int(u*size),size-1)])
                else:
                    post=posterior(history,p['inference'],129);folder=route/f'round-{round_number}-selector';folder.mkdir()
                    context=context_for(case,route,size,round_number,history,post,classical);core.validate_context(context)
                    core.write(route/f'round-{round_number}-context-compute.json',{'posterior_elapsed_s':post['elapsed_s'],'classical_elapsed_s':classical_elapsed})
                    core.write(folder/'context.json',context);(folder/'instructions.txt').write_text(INSTRUCTIONS)
                    contexts.append({'case_id':spec['case_id'],'menu_size':size,'context':str(folder/'context.json'),'instructions':str(folder/'instructions.txt')});continue
                core.write(route/f'round-{round_number}-decision.json',{'candidate_id':choice,'policy':policy,'frozen_at_utc':core.now()})
    core.write(root/f'round-{round_number}-contexts.json',contexts)
    frozen={str(p.relative_to(root)):core.sha(p) for p in root.rglob(f'round-{round_number}-*') if p.is_file()}
    for p in root.rglob(f'round-{round_number}-selector/*'):
        if p.is_file():frozen[str(p.relative_to(root))]=core.sha(p)
    core.write(root/f'round-{round_number}-frozen.json',{'artifacts_sha256':frozen})
    core.trace(root,'decision',f'Round{round_number} numerical/fixed/random choices and unscored LLM contexts frozen before fresh measurements.',[root/f'round-{round_number}-contexts.json',root/f'round-{round_number}-frozen.json'])
    print(f'CONTEXTS={root/f"round-{round_number}-contexts.json"}',flush=True)


def prepare_initial(root):
    source_check(root);core.check_manifest(root,'input-manifest.json')
    for spec in core.read(root/'protocol.json')['cases']:
        case=root/'cases'/spec['case_id'];job=case/'baseline'
        command=[sys.executable,str(case/'sources/experiments/hybrid_selection_round.py'),'worker','--campaign',str(case),'--job','baseline']
        status=launch(case,job,command);core.trace(case,'evidence','Shared initial neural fit retained for initial-only diagnostics; final models will start from seed0.',list(job.iterdir()))
        if status['status']!='worker_finished':raise RuntimeError('Initial fit failed; retain failed study, no selector contexts manufactured')
    freeze_round(root,1)


def validate_map(root,round_number,map_path):
    entries=core.read(map_path);expected={(e['case_id'],e['menu_size']) for e in core.read(root/f'round-{round_number}-contexts.json')}
    if not isinstance(entries,list) or len(entries)!=8 or {(e['case_id'],e['menu_size']) for e in entries}!=expected:raise ValueError('exactly8 case/menu decisions required')
    for entry in entries:
        case=root/'cases'/entry['case_id'];route=route_path(case,entry['menu_size'],'fresh_llm');folder=route/f'round-{round_number}-selector'
        context=core.read(folder/'context.json');core.parse_decision(Path(entry['decision']).read_text(),core.validate_context(context))
        if not isinstance(core.read(entry['metadata']),dict):raise ValueError('metadata must be an object')
    return entries


def acquire(root,round_number,map_path):
    if (root/f'round-{round_number}-lock.json').exists():raise ValueError('acquisition round already started; no retries')
    if not (root/f'round-{round_number}-contexts.json').exists():raise ValueError('acquisition contexts not prepared')
    if round_number==2:
        if not (root/'round-1-complete.json').exists():raise ValueError('round1 must complete before round2')
        core.check_manifest(root,'round-1-complete.json')
    source_check(root);core.check_manifest(root,'input-manifest.json');core.check_manifest(root,f'round-{round_number}-frozen.json')
    entries=validate_map(root,round_number,map_path);core.write(root/f'round-{round_number}-lock.json',{'started_at_utc':core.now()})
    incoming=root/f'round-{round_number}-incoming';incoming.mkdir();shutil.copyfile(map_path,root/f'round-{round_number}-decision-map.json')
    for entry in entries:
        case=root/'cases'/entry['case_id'];route=route_path(case,entry['menu_size'],'fresh_llm')
        shutil.copyfile(entry['decision'],route/f'round-{round_number}-decision.json');shutil.copyfile(entry['metadata'],route/f'round-{round_number}-metadata.json')
    core.trace(root,'decision',f'All8 actual round{round_number} model responses validated/preserved before any evaluator measurements.',[root/f'round-{round_number}-decision-map.json'])
    for spec in core.read(root/'protocol.json')['cases']:
        case=root/'cases'/spec['case_id'];p=core.read(case/'protocol.json')
        for size in MENUS:
            for policy in POLICIES:
                route=route_path(case,size,policy);choice=core.read(route/f'round-{round_number}-decision.json')['candidate_id']
                actions={candidate_id(u):u for u in MENUS[size]}
                if choice not in actions:raise ValueError('policy chose unknown menu action')
                old=core.read(case/'baseline/observations.json') if round_number==1 else core.read(route/'round-1-observations.json')
                new=core.serialize(core.generate(p,actions[choice],spec['candidate_noise_seeds'][str(round_number)][choice]))
                core.write(route/f'round-{round_number}-new-observations.json',new);core.write(route/f'round-{round_number}-observations.json',old+[new])
                started=time.monotonic();classical=fit_classical(old+[new],p)
                core.write(route/f'round-{round_number}-after-classical.json',{'fit':classical,'elapsed_s':time.monotonic()-started})
                core.trace(case,'action',f'menu{size}/{policy} round{round_number} selected{choice}; evaluator noise keyed by case/round/action.',[route/f'round-{round_number}-decision.json',route/f'round-{round_number}-observations.json',route/f'round-{round_number}-after-classical.json'])
    completed={str(p.relative_to(root)):core.sha(p) for p in root.rglob(f'round-{round_number}-*') if p.is_file()}
    core.write(root/f'round-{round_number}-complete.json',{'completed_at_utc':core.now(),'artifacts_sha256':completed})
    if round_number==1:freeze_round(root,2)
    else:train_all_final(root)


def train_final(root,case_id,size,policy):
    case=root/'cases'/case_id;core.source_check(case);route=route_path(case,size,policy);p=core.read(case/'protocol.json')
    observations=[core.deserialize(o) for o in core.read(route/'round-2-observations.json')]
    if len(observations)!=3 or sum(len(o.times) for o in observations)!=36:raise ValueError('final measurement budget must be36')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=core.BallPINN(observations,p['inference'],p['pinn'],seed=0);state=model.state_dict()
    core.write(route/'initial-state.json',{'seed':0,'starting_state_sha256':core.state_digest(state,[k for k in state if k!='tau_bounds']),
                                        'training_from':'Fresh identical seed, not baseline warm start','observation_count':36})
    try:
        with (route/'training.jsonl').open('x') as log:
            def progress(row):log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            training=core.train(model,progress)
        torch.save({'model_state_dict':model.state_dict()},route/'checkpoint.pt')
        result={'status':'complete','parameters':{'floor_m':float(model.floor.detach()),'restitution':float(model.restitution.detach())},
                'training':training,'diagnostics':model.diagnostics(197),'evaluation':core.evaluate(model,observations,p),
                'classical':core.read(route/'round-2-after-classical.json')['fit'],'completed_at_utc':core.now()}
        core.write(route/'result.json',result)
    except Exception:
        torch.save({'model_state_dict':model.state_dict()},route/'failed-checkpoint.pt');(route/'worker-stack.txt').write_text(traceback.format_exc());raise


def summarize(rows):
    if len(rows)!=4 or not all(r['complete'] for r in rows):return {'status':'unavailable_incomplete','means':None,'interactions':None,'reason':'Missing/failed route suppresses all aggregate ranking, no successful-subset conclusion.'}
    means={str(size):{p:float(np.mean([r['menus'][str(size)][p]['evaluation'][PRIMARY] for r in rows])) for p in POLICIES} for size in MENUS}
    classical={str(size):{p:float(np.mean([r['menus'][str(size)][p]['classical']['evaluation'][PRIMARY] for r in rows])) for p in POLICIES} for size in MENUS}
    interactions={}
    for comparator in ['posterior_numerical','fixed_downward','fixed_upward','uniform_random']:
        values=[(r['menus']['9']['fresh_llm']['evaluation'][PRIMARY]-r['menus']['9'][comparator]['evaluation'][PRIMARY])-(r['menus']['3']['fresh_llm']['evaluation'][PRIMARY]-r['menus']['3'][comparator]['evaluation'][PRIMARY]) for r in rows]
        interactions[comparator]={'values':values,'mean':float(np.mean(values)),'minimum':min(values),'maximum':max(values),
                                 'definition':'(LLM−comparator)menu9−(LLM−comparator)menu3; negative favors menu-expansion interaction, descriptive only.'}
    return {'status':'complete_descriptive_only','means':means,'classical_means':classical,'interactions':interactions,
            'interpretation':'Four physical cases, one noise stream each, sequential one-step acquisition. Candidate/task complexity differs; model size does not. Equal samples/finaltraining are not equal total compute.'}


def collect(root):
    rows=[]
    for spec in core.read(root/'protocol.json')['cases']:
        case=root/'cases'/spec['case_id'];row={**spec,'complete':True,'menus':{},'baseline':core.read(case/'baseline/result.json')}
        for size in MENUS:
            row['menus'][str(size)]={}
            for policy in POLICIES:
                route=route_path(case,size,policy);p=route/'result.json';process=route/'process.json'
                success=p.exists() and process.exists() and core.read(process)['status']=='worker_finished'
                if not success:row['complete']=False
                fit=core.read(p) if p.exists() else {'status':'unavailable','process':core.read(process) if process.exists() else None}
                fit['choices']=[core.read(route/f'round-{r}-decision.json')['candidate_id'] for r in [1,2]]
                fit['classical_by_round']=[core.read(route/f'round-{r}-after-classical.json') for r in [1,2]]
                if (route/'round-1-numerical.json').exists():fit['numerical_design']=[core.read(route/f'round-{r}-numerical.json') for r in [1,2]]
                row['menus'][str(size)][policy]=fit
        rows.append(row)
    return rows


def train_all_final(root):
    for spec in core.read(root/'protocol.json')['cases']:
        case=root/'cases'/spec['case_id']
        for size in MENUS:
            for policy in POLICIES:
                route=route_path(case,size,policy)
                command=[sys.executable,str(root/'sources'/SELF),'worker-final','--campaign',str(root),'--case',spec['case_id'],'--menu',str(size),'--policy',policy]
                launch(case,route,command);core.trace(case,'evidence',f'menu{size}/{policy} final36-position PINN retained under equal full budgets/fromseed0.',list(route.glob('*.json'))+list(route.glob('*.pt'))+[route/'stdout.log',route/'stderr.log'])
            print(f'FINAL_FITS={spec["case_id"]}/menu{size}',flush=True)
        core.write(case/'manifest.json',{'compute_elapsed_s':sum(core.read(p)['elapsed_s'] for p in processes(case)),'artifacts_sha256':core.inventory(case,{'manifest.json'})})
    rows=collect(root);summary=summarize(rows)
    policy_compute={'numerical_design_s':sum(core.read(p)['compute_elapsed_s'] for p in root.glob('cases/*/routes/*/*/round-*-numerical.json')),
                    'classical_fitting_s':sum(core.read(p)['elapsed_s'] for p in root.glob('cases/*/routes/*/*/round-*-classical.json')),
                    'llm_context_posterior_s':sum(core.read(p)['posterior_elapsed_s'] for p in root.glob('cases/*/routes/*/*/round-*-context-compute.json')),
                    'selector_metadata':[core.read(p) for p in root.glob('cases/*/routes/*/fresh_llm/round-*-metadata.json')],
                    'completed_at_utc':core.now(),'started_at_utc':core.read(root/'environment.json')['started_at_utc']}
    core.write(root/'results.json',{'cases':rows,'aggregate':summary,'policy_compute':policy_compute})
    lines=['# Candidate-menu expansion/two-step pilot','',summary['interpretation'] if summary['means'] else summary['reason'],'',
           '|Menu|Policy|Mean final PINN transfer RMSE|Mean final classical transfer RMSE|','|---:|---|---:|---:|']
    if summary['means']:
        for size in MENUS:
            for policy in POLICIES:lines.append(f"|{size}|{policy}|{summary['means'][str(size)][policy]:.6f}|{summary['classical_means'][str(size)][policy]:.6f}|")
        lines+=['','|LLM interaction against|Mean|Minimum|Maximum|','|---|---:|---:|---:|']
        for comparator,item in summary['interactions'].items():lines.append(f"|{comparator}|{item['mean']:.6f}|{item['minimum']:.6f}|{item['maximum']:.6f}|")
    lines+=['','Per-case paths, both classical acquisition rounds, all posterior/EIG refinement diagnostics, MCSE/paired gaps, full neural diagnostics and timing are retained. LLM contexts include raw history/known physics/prior/uncertainty but no prescribed utilities/rankings/truth/reservedscores.', '',
            'Primary errors are analytic parameter-transfer position RMSE on reserved inputs, not direct neural generalization. All final PINNs train from identicalseed0/fullbudgets, with36 observedpositions. Initial PINN diagnostics are labeled initial-only and stale after first acquisition.', '',
            'This is a small candidate/task-complexity experiment, not model scaling, real-world validation or novelty proof. No tuning, retries or favorable-subset analysis. Numerical posterior resolution/MC limitations and unequal policy compute must be considered. All raw actual modelresponses and unavailableusage annotations are retained.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    core.trace(root,'next_step','Assess matched menu interaction against uncertainty-aware numerical and both fixed directions without claiming scalegain from a small descriptive study.',[root/'results.json',root/'report.md'])
    allprocess=[p for spec in core.read(root/'protocol.json')['cases'] for p in processes(root/'cases'/spec['case_id'])]
    counts={s:sum(core.read(p)['status']==s for p in allprocess) for s in sorted({core.read(p)['status'] for p in allprocess})}
    core.write(root/'manifest.json',{'status':'complete' if summary['means'] else 'completed_with_failures','pinn_compute_elapsed_s':sum(core.read(p)['elapsed_s'] for p in allprocess),
               'training_process_status_counts':counts,'artifacts_sha256':core.inventory(root,{'manifest.json'})})
    verify(root);print(f'COMPLETE={root}',flush=True)


def verify(root):
    source_check(root)
    if not (root/'manifest.json').exists():
        core.check_manifest(root,'input-manifest.json')
        for r in [1,2]:
            if (root/f'round-{r}-frozen.json').exists():core.check_manifest(root,f'round-{r}-frozen.json')
        print('Scale pilot frozen inputs/contexts verified');return
    manifest=core.check_manifest(root,'manifest.json')
    if set(manifest['artifacts_sha256'])!=set(core.inventory(root,{'manifest.json'})):raise ValueError('scale inventory mismatch')
    if manifest['status']=='failed':
        print('Failed scale pilot artifacts retained and verified; aggregate unavailable');return
    for spec in core.read(root/'protocol.json')['cases']:
        case=root/'cases'/spec['case_id'];m=core.check_manifest(case,'manifest.json')
        if m['compute_elapsed_s']>150 or len(processes(case))>11:raise ValueError('case budget exceeded')
        for row in map(json.loads,(case/'thoughts.jsonl').read_text().splitlines()):
            for name,digest in row['artifacts_sha256'].items():
                if m['artifacts_sha256'].get(name)!=digest:raise ValueError('case trace mismatch')
    for row in map(json.loads,(root/'thoughts.jsonl').read_text().splitlines()):
        for name,digest in row['artifacts_sha256'].items():
            if manifest['artifacts_sha256'].get(name)!=digest:raise ValueError('root trace mismatch')
    if manifest['pinn_compute_elapsed_s']>600 or sum(manifest['training_process_status_counts'].values())>44:raise ValueError('global budget exceeded')
    print('Scale pilot sources, inputs, artifacts, traces and budgets verified')


def failure_manifest(root,error):
    if (root/'manifest.json').exists():return
    (root/'controller-stack.txt').write_text(traceback.format_exc())
    core.write(root/'failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'completed_at_utc':core.now()})
    core.write(root/'results.json',{'aggregate':{'status':'unavailable_incomplete','means':None,'reason':'Stage failure; no successful-subset ranking.'}})
    (root/'report.md').write_text('Study failed before complete routes. Partial data, decisions, logs and stack retained; no retry or aggregate ranking.\n')
    core.write(root/'manifest.json',{'status':'failed','artifacts_sha256':core.inventory(root,{'manifest.json'})})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','prepare-initial','acquire','worker-final','verify'])
    parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path);parser.add_argument('--round',type=int,choices=[1,2]);parser.add_argument('--decision-map',type=Path)
    parser.add_argument('--case');parser.add_argument('--menu',type=int,choices=[3,9]);parser.add_argument('--policy',choices=POLICIES)
    args=parser.parse_args()
    preexisting=args.out.exists() if args.command=='prepare' and args.out else False
    try:
        if args.command=='prepare':prepare(args.out.resolve())
        elif args.command=='prepare-initial':prepare_initial(args.campaign.resolve())
        elif args.command=='acquire':acquire(args.campaign.resolve(),args.round,args.decision_map)
        elif args.command=='worker-final':train_final(args.campaign.resolve(),args.case,args.menu,args.policy)
        else:verify(args.campaign.resolve())
    except Exception as error:
        root=args.out if args.command=='prepare' else args.campaign
        record_stage=(args.command=='prepare' and not preexisting) or args.command=='prepare-initial' or (args.command=='acquire' and (root/f'round-{args.round}-lock.json').exists() and not isinstance(error,ValueError))
        if root and root.exists() and record_stage:
            failure_manifest(root,error)
        raise


if __name__=='__main__':main()
