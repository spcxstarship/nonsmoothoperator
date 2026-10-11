"""Timing-only design exploration, not confirmation or a PINN training campaign.

Run: .venv/bin/python experiments/hybrid_timing_audit.py run --out data/research/NEW
Verify: .venv/bin/python NEW/sources/hybrid_timing_audit.py verify --campaign NEW
Exactly25 declared shared12-time schedules, full one-impact prior, fixed inputs.
No release-height exploration, adaptive tuning, LLM selector or training is run.
"""
import argparse
from datetime import datetime,timezone
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

CANDIDATES=[('downward',-4.),('neutral',0.),('upward',4.)]
BOUNDS=np.array([[0.,.25],[.6,.95]])
ORIGINAL=[.08,.14,.20,.26,.32,.38,.44,.50,.56,.62,.68,.70]


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def read(path):return json.loads(Path(path).read_text())


def schedules():
    rows=[{'id':'original','family':'original','times_s':ORIGINAL}]
    def add(identifier,family,times):rows.append({'id':identifier,'family':family,'times_s':np.asarray(times).tolist()})
    for start in [.01,.08,.20,.35]:add(f'uniform-{start:.2f}','uniform',np.linspace(start,.7,12))
    for start in [.44,.50,.56,.59,.64,.66]:add(f'late-{start:.2f}','late',np.linspace(start,.7,12))
    for end in [.20,.40,.45]:add(f'early-{end:.2f}','early',np.r_[np.linspace(.02,end,11),.7])
    add('early-parent-0.35','early',np.r_[np.linspace(.05,.35,11),.7])
    for early_count in [3,6,9]:
        for late_start in [.50,.60]:
            add(f'mixed-{early_count}-{late_start:.2f}','mixed',np.r_[np.linspace(.02,.30,early_count),np.linspace(late_start,.7,12-early_count)])
    for center in [.36,.42,.55,.60]:add(f'cluster-{center:.2f}','clustered',np.r_[np.linspace(center-.04,center+.04,11),.7])
    return rows


def validate_schedule(times):
    t=np.asarray(times)
    if t.shape!=(12,) or np.any(~np.isfinite(t)) or t[0]<0 or abs(t[-1]-.7)>1e-12 or np.any(np.diff(t)<.001):
        raise ValueError('schedule requires12 increasing times, >=1ms separation, final0.7s')
    for _,u in CANDIDATES:
        for floor in BOUNDS[0]:
            tau=np.sqrt(2*(1-floor)/(9.81-u))
            if np.sqrt(2/(9.81-u))>=t[-1]:raise ValueError('latest prior first impact not observed')
            for restitution in BOUNDS[1]:
                if t[-1]>=(1+2*restitution)*tau:raise ValueError('second impact within schedule')


def position(parameters,u,times):
    """Broadcast analytic one-rebound world height over parameter rows and times."""
    parameters=np.atleast_2d(parameters);t=np.asarray(times)[None,:]
    floor=parameters[:,0,None];e=parameters[:,1,None];a=9.81-u
    tau=np.sqrt(2*(1-floor)/a);elapsed=t-tau
    return np.where(t<tau,1-.5*a*t*t,floor+e*a*tau*elapsed-.5*a*elapsed*elapsed)


def jacobian(parameters,u,times,step=1e-5):
    """Finite derivatives in dimensionless prior coordinates, scaled by noise0.01."""
    p=np.atleast_2d(parameters);q=(p-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0])
    columns=[]
    for axis in [0,1]:
        left=q.copy();right=q.copy();left[:,axis]=np.maximum(0,q[:,axis]-step);right[:,axis]=np.minimum(1,q[:,axis]+step)
        lp=BOUNDS[:,0]+left*(BOUNDS[:,1]-BOUNDS[:,0]);rp=BOUNDS[:,0]+right*(BOUNDS[:,1]-BOUNDS[:,0])
        columns.append((position(rp,u,times)-position(lp,u,times))/(right[:,axis]-left[:,axis])[:,None]/.01)
    return np.stack(columns,axis=-1)


def analytic_jacobian(parameters,u,times):
    p=np.atleast_2d(parameters);t=np.asarray(times)[None,:];a=9.81-u
    tau=np.sqrt(2*(1-p[:,0,None])/a);post=t>=tau
    floor=np.where(post,(1+p[:,1,None])*(2-t/tau)*.25/.01,0.)
    restitution=np.where(post,a*tau*(t-tau)*.35/.01,0.)
    return np.stack([floor,restitution],axis=-1)


def grid(resolution):
    floor,e=np.meshgrid(np.linspace(*BOUNDS[0],resolution),np.linspace(*BOUNDS[1],resolution),indexing='ij')
    return np.column_stack([floor.ravel(),e.ravel()])


def scores(parameters,times,step=1e-5):
    initial=jacobian(parameters,0.,times,step)
    f0=np.einsum('nki,nkj->nij',initial,initial)
    output=[]
    for _,u in CANDIDATES:
        j=jacobian(parameters,u,times,step)
        information=f0+np.einsum('nki,nkj->nij',j,j)+1e-9*np.eye(2)
        sign,value=np.linalg.slogdet(information)
        if np.any(sign<=0):raise FloatingPointError('nonpositive information determinant')
        output.append(value)
    return np.column_stack(output)


def describe(parameters,score,gap_min):
    order=np.argsort(-score,axis=1,kind='stable');winner=order[:,0];gap=score[np.arange(len(score)),winner]-score[np.arange(len(score)),order[:,1]]
    witnesses=[]
    for index,(identifier,_) in enumerate(CANDIDATES):
        available=np.flatnonzero((winner==index)&(gap>=gap_min))
        if len(available):
            chosen=available[np.argmax(gap[available])]
            witnesses.append({'candidate_id':identifier,'parameters':parameters[chosen].tolist(),'scores':score[chosen].tolist(),'gap':float(gap[chosen])})
    return {'winner_counts':{identifier:int(np.sum(winner==i)) for i,(identifier,_) in enumerate(CANDIDATES)},
            'robust_witnesses':witnesses,'has_multiple_robust_winners':len(witnesses)>=2,
            'winner_index':winner.tolist(),'top_gap':gap.tolist()}


def trace(root,kind,summary,artifacts):
    with (root/'thoughts.jsonl').open('a') as stream:
        stream.write(json.dumps({'timestamp_utc':now(),'kind':kind,'summary':summary,
                                'artifacts_sha256':{str(p.relative_to(root)):sha(p) for p in artifacts},
                                'usage':{'source':'No LLM invocation; model/token usage is unavailable/not applicable','model_id':None,'input_tokens':None,'output_tokens':None}})+'\n')


def inventory(root):return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}


def run(root):
    root.mkdir(parents=True,exist_ok=False);sources=root/'sources';sources.mkdir()
    shutil.copyfile(__file__,sources/'hybrid_timing_audit.py')
    declared=schedules()
    for schedule in declared:validate_schedule(schedule['times_s'])
    protocol={'study':'predeclared timing-only design exploration','frozen_at_utc':now(),'schedules':declared,
              'floor_prior':[0.,.25],'restitution_prior':[.6,.95],'release_height':1.,'initial_input':0.,
              'candidates':CANDIDATES,'gravity':9.81,'noise_std':.01,'measurements':12,'horizon':.7,
              'primary_grid_resolution':17,'refinement_grid_resolution':33,'finite_difference_step':1e-5,'refined_derivative_step':5e-6,
              'jitter':1e-9,'tie_rule':'listed candidate order','nontrivial_logdet_gap':.05,
              'timing_perturbation_s':1e-5,'parameter_perturbation_normalized':1e-4,
              'perturbation_rule':'Each witness tested at original schedule and plus/minus10us interior times (last stays0.7), and plus/minus1e-4 in each normalized parameter coordinate, clipped to prior.',
              'maximum_compute_s':300,'command_timeout_s':120,'frozen_source_sha256':sha(sources/'hybrid_timing_audit.py'),
              'stopping_rule':'Exactly25 frozen schedules; no tuning/extensions or PINN fits. Positive audit is exploration, not confirmation.'}
    write(root/'protocol.json',protocol)
    write(root/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.system(),'invocation':sys.argv})
    trace(root,'hypothesis','Changing shared measurement timing might alter the informative actuation across the prior; a negative audit is a valid stopping outcome.',[root/'protocol.json',root/'environment.json',sources/'hybrid_timing_audit.py'])
    command=[sys.executable,str(sources/'hybrid_timing_audit.py'),'evaluate','--campaign',str(root)]
    write(root/'invocation.json',{'argv':command,'timeout_s':120,'started_at_utc':now()})
    started=time.monotonic()
    with (root/'stdout.log').open('x') as stdout,(root/'stderr.log').open('x') as stderr:
        try:
            subprocess.run(command,stdout=stdout,stderr=stderr,timeout=120,check=True)
            status='complete'
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as error:
            status='failed';(root/'controller-stack.txt').write_text(traceback.format_exc())
    elapsed=time.monotonic()-started
    write(root/'process.json',{'status':status,'elapsed_s':elapsed,'completed_at_utc':now()})
    trace(root,'evidence','Frozen audit process finished; all score grids, refinements and failure logs remain retained.',[root/'process.json',root/'stdout.log',root/'stderr.log']+([root/'findings.json',root/'report.md'] if (root/'findings.json').exists() else []))
    write(root/'manifest.json',{'status':status,'elapsed_s':elapsed,'artifacts_sha256':inventory(root)})
    verify(root)
    print(f'Complete timing audit: {root}',flush=True)


def perturb_checks(witness,times,protocol):
    p=np.array(witness['parameters']);q=(p-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0]);expected=[c[0] for c in CANDIDATES].index(witness['candidate_id'])
    rows=[]
    variants=[('original',p)]
    for axis in [0,1]:
        for sign in [-1,1]:
            changed=q.copy();changed[axis]=np.clip(changed[axis]+sign*1e-4,0,1)
            variants.append((f'parameter-{axis}-{sign}',BOUNDS[:,0]+changed*(BOUNDS[:,1]-BOUNDS[:,0])))
    for name,param in variants:
        for sign in [-1,0,1]:
            t=np.asarray(times).copy();t[:-1]+=sign*1e-5;validate_schedule(t)
            s=scores(param,t,step=protocol['refined_derivative_step'])[0]
            order=np.argsort(-s,kind='stable');gap=float(s[order[0]]-s[order[1]])
            rows.append({'parameter_variant':name,'timing_sign':sign,'winner':CANDIDATES[order[0]][0],'gap':gap,
                         'passes':int(order[0])==expected and gap>=protocol['nontrivial_logdet_gap']})
    return {'all_pass':all(row['passes'] for row in rows),'checks':rows}


def evaluate(root):
    protocol=read(root/'protocol.json')
    if sha(Path(__file__))!=protocol['frozen_source_sha256']:raise ValueError('executed source differs from frozen source')
    started=time.monotonic();results=[]
    for schedule in protocol['schedules']:
        if time.monotonic()-started>=300:raise TimeoutError('audit total compute budget exhausted')
        times=schedule['times_s'];record={'id':schedule['id'],'family':schedule['family'],'times_s':times}
        for name,resolution in [('primary',17),('refinement',33)]:
            parameters=grid(resolution);score=scores(parameters,times)
            description=describe(parameters,score,.05)
            write(root/f"{schedule['id']}-{name}.json",{'parameters':parameters.tolist(),'scores':score.tolist(),**description})
            record[name]={k:v for k,v in description.items() if k not in ['winner_index','top_gap']}
        witnesses=record['refinement']['robust_witnesses']
        record['perturbations']=[{'candidate_id':w['candidate_id'],'parameters':w['parameters'],**perturb_checks(w,times,protocol)} for w in witnesses]
        surviving={p['candidate_id'] for p in record['perturbations'] if p['all_pass']}
        record['viable_switch']=record['primary']['has_multiple_robust_winners'] and len(surviving)>=2
        results.append(record)
        print(json.dumps({'schedule':schedule['id'],'refined_winners':record['refinement']['winner_counts'],'viable_switch':record['viable_switch']}),flush=True)
    positive=[r['id'] for r in results if r['viable_switch']]
    findings={'study_type':'design exploration, not confirmation','viable_schedules':positive,'schedule_results':results,'evaluation_elapsed_s':time.monotonic()-started,
              'conclusion':'At least one frozen timing family supports robust parameter-dependent candidate choice; requires a separate frozen validation design.' if positive else 'No robust choice switch among the25 declared timing schedules; adaptation is not established by this timing-only audit.'}
    write(root/'findings.json',findings)
    lines=['# Timing design audit','',findings['conclusion'],'',
           'No PINN fits or release-height extensions were run. All25 schedules,17×17 primary and33×33 refinement grids, gap threshold0.05 and tiny perturbations were frozen before outcomes. Truth/prior grids are evaluator-only and must never be copied into later selector contexts.', '',
           '|Schedule|Family|Refined downward / neutral / upward counts|Robust switch after checks|','|---|---|---|---|']
    for record in results:
        counts=record['refinement']['winner_counts'];lines.append(f"|{record['id']}|{record['family']}|{counts['downward']} / {counts['neutral']} / {counts['upward']}|{record['viable_switch']}|")
    lines+=['','A viable switch requires at least two candidate winners with logdet gap≥0.05 on both declared grids and surviving15 timing/parameter perturbations at each refined winner witness. Full scores and gaps are retained for every grid point. These are local information predictions, not measured learning benefit.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def verify(root):
    manifest=read(root/'manifest.json')
    if set(manifest['artifacts_sha256'])!=set(inventory(root)):raise ValueError('artifact inventory mismatch')
    for name,digest in manifest['artifacts_sha256'].items():
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or sha(p)!=digest:raise ValueError(f'artifact hash mismatch: {name}')
    protocol=read(root/'protocol.json')
    if sha(root/'sources/hybrid_timing_audit.py')!=protocol['frozen_source_sha256']:raise ValueError('source hash mismatch')
    for row in map(json.loads,(root/'thoughts.jsonl').read_text().splitlines()):
        for name,digest in row['artifacts_sha256'].items():
            if manifest['artifacts_sha256'].get(name)!=digest:raise ValueError('trace artifact mismatch')
    if manifest['elapsed_s']>300:raise ValueError('audit budget violation')
    print('Timing audit sources, artifacts and trace verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','evaluate','verify']);parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path)
    args=parser.parse_args()
    if args.command=='run':run(args.out.resolve())
    elif args.command=='evaluate':evaluate(args.campaign.resolve())
    else:verify(args.campaign.resolve())


if __name__=='__main__':main()
