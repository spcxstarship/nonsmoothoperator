"""Original synthetic stick-slip/impact class-identification pilot, not paper replication.
prepare --out NEW; acquire --campaign DIR --round N --decision-map JSON;
verify --campaign DIR. Execute the frozen sources/experiments/stick_slip_pilot.py.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
SELF='experiments/stick_slip_pilot.py'
CLASSES=['coulomb_constant','coulomb_speed','static_constant','static_speed']
POLICIES=['fresh_llm','class_information','fixed_balanced','uniform_random']
PROFILES={'breakaway':[.18,.26,.34,.42],'gentle_reversal':[.30,-.30,.30,-.30],
          'impact_reversal':[1.2,-1.2,1.2,-1.2],'high_speed':[2.,-2.,2.,-2.],
          'asymmetric':[.5,1.5,-.4,-1.5],'release_rest':[1.,0.,-1.,0.],
          'threshold_steps':[.20,.28,-.20,-.28],'short_push':[1.6,0.,0.,-.3],
          'rising_reversal':[.4,.8,-1.2,-1.6],'alternating_strength':[.35,-1.4,.35,-1.4]}
BASELINE=[.12]*4
FIXED=['breakaway','impact_reversal','high_speed']
TIMES=np.linspace(.15,4.,16)
NOISE=.003
INSTRUCTIONS='''Choose ONE next force-profile experiment to distinguish the four proposed contact/friction classes and improve prediction. Use only observed noisy positions, declared model primitives/prior and available profile descriptions. Three sequential choices total; future observations are unavailable. Return ONLY JSON with candidate_id, hypothesis, rationale. Candidate must be an exact supplied ID. Nonempty text fields under700characters. No changes to sample times/model/measurement count.'''

def now():return datetime.now(timezone.utc).isoformat()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(root,excluded=()):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and str(p.relative_to(root)) not in excluded}
def trace(root,kind,summary,paths=()):
    with (root/'trace.jsonl').open('a') as f:f.write(json.dumps({'time':now(),'kind':kind,'summary':summary,'usage':None,'usage_source':'unavailable from local execution/collaboration','artifacts':{str(p.relative_to(root)):sha(p) for p in paths}})+'\n')

def bank():
    return [{'class':label,'friction':f,'restitution':e,'spring':k,'damping':c} for label in CLASSES for f in [.16,.22] for e in [.72,.82] for k,c in [(2.,.14),(2.4,.22)]]

def slip(t,x,v,force,p,direction):
    t=np.asarray(t);a=p['damping']/2;w=np.sqrt(p['spring']-a*a);eq=(force-direction*p['friction'])/p['spring'];b=x-eq;d=(v+a*b)/w
    exp=np.exp(-a*t);co=np.cos(w*t);si=np.sin(w*t)
    return eq+exp*(b*co+d*si),exp*((-a*b+w*d)*co+(-a*d-w*b)*si)

def simulate(p,profile,times=TIMES,collapse_speed=1e-6,root_tolerance=1e-12):
    """Exact damped linear slip segments with analytic velocity-zero time and monotone wall root.
    Stick releases at force discontinuities. No penetration clipping or hidden chatter fix.
    """
    fs=p['friction']*(1.6 if p['class'].startswith('static') else 1.)
    x=v=0.;segments=[];events=[];count=0
    for interval,force in enumerate(profile):
        t=float(interval);end=float(interval+1)
        while t<end-1e-12:
            count+=1
            if count>300:raise RuntimeError('event/chatter cap exceeded')
            net=force-p['spring']*x
            if abs(v)<1e-10:
                atwall=abs(x)>=.25-1e-9
                if atwall and x*net>=-abs(x)*fs:
                    segments.append((t,end,x,0.,force,0));events.append({'time':t,'kind':'wall_rest','position':x});t=end;continue
                if abs(net)<=fs+1e-12:
                    segments.append((t,end,x,0.,force,0));events.append({'time':t,'kind':'stick','position':x});t=end;continue
                direction=float(np.sign(net));events.append({'time':t,'kind':'release','position':x})
            else:direction=float(np.sign(v))
            duration=end-t;alpha=p['damping']/2;omega=np.sqrt(p['spring']-alpha*alpha);eq=(force-direction*p['friction'])/p['spring']
            angle=float(np.arctan2(-v,(-alpha*v-p['spring']*(x-eq))/omega)%np.pi)
            if angle<1e-10:angle+=np.pi
            stop=angle/omega;dt=min(duration,stop);kind='velocity' if stop<=duration else None
            endpoint=float(slip(dt,x,v,force,p,direction)[0])
            if endpoint>=.25 or endpoint<=-.25:
                wall=.25 if endpoint>=.25 else -.25
                dt=brentq(lambda z:float(slip(z,x,v,force,p,direction)[0])-wall,0.,dt,xtol=root_tolerance)
                kind='right' if wall>0 else 'left'
            if dt<1e-9:raise RuntimeError('nonadvancing event/chatter')
            segments.append((t,t+dt,x,v,force,direction));xnew,vnew=slip(dt,x,v,force,p,direction);x=float(xnew);v=float(vnew);t+=dt
            if kind in ['left','right']:
                e=p['restitution']-.10*abs(v) if p['class'].endswith('speed') else p['restitution'];e=float(np.clip(e,.2,.95))
                events.append({'time':t,'kind':kind,'position':x,'incoming_velocity':v,'restitution':e});x=.25 if kind=='right' else -.25;v=-e*v
                if abs(v)<collapse_speed:
                    events.append({'time':t,'kind':'low_speed_contact_collapse','position':x});v=0.
            elif kind=='velocity':
                events.append({'time':t,'kind':'velocity_zero','position':x});v=0.
    out=[]
    for value in times:
        segment=next((s for s in segments if s[0]-1e-11<=value<=s[1]+1e-11),None)
        if segment is None:raise ValueError('sample outside simulation horizon')
        a,b,x,v,force,direction=segment
        position=x if direction==0 else float(slip(value-a,x,v,force,p,direction)[0]);out.append(position)
    if max(np.abs(out))>.25000001:raise RuntimeError('penetration detected')
    return np.array(out),events

def preflight():
    summaries={}
    for name,profile in {'baseline':BASELINE,**PROFILES}.items():
        rows=[]
        for p in bank():
            y,events=simulate(p,profile)
            rows.append({kind:sum(e['kind']==kind for e in events) for kind in ['left','right','stick','release','velocity_zero']})
        summaries[name]={'min':{k:min(r[k] for r in rows) for k in rows[0]},'max':{k:max(r[k] for r in rows) for k in rows[0]}}
    return summaries

def posterior(history,predictions,particles):
    logw=np.zeros(len(particles))
    for obs in history:logw-=np.sum((predictions[obs['candidate_id']]-np.array(obs['positions'])[None,:])**2,axis=1)/(2*NOISE**2)
    logw-=logsumexp(logw);w=np.exp(logw)
    probabilities={label:float(sum(w[i] for i,p in enumerate(particles) if p['class']==label)) for label in CLASSES}
    return w,probabilities

def class_information(w,predictions,particles,seed,draws=128):
    rng=np.random.default_rng(seed);indices=np.searchsorted(np.cumsum(w),rng.random(draws));indices=np.minimum(indices,len(w)-1);noise=rng.normal(size=(draws,len(TIMES)));labels=np.array([p['class'] for p in particles]);scores={};contributions={}
    logw=np.full(len(w),-np.inf);np.log(w,out=logw,where=w>0)
    for candidate in PROFILES:
        means=predictions[candidate];y=means[indices]+NOISE*noise
        likelihood=-.5*np.sum((y[:,None,:]-means[None,:,:])**2,axis=2)/NOISE**2
        mix=logsumexp(logw[None,:]+likelihood,axis=1);values=[]
        for i,index in enumerate(indices):
            same=labels==labels[index];mass=w[same].sum();values.append(logsumexp(logw[same]+likelihood[i,same])-np.log(mass)-mix[i])
        values=np.array(values);contributions[candidate]=values.tolist();scores[candidate]={'mean':float(values.mean()),'mcse':float(values.std(ddof=1)/np.sqrt(draws))}
    order=sorted(PROFILES,key=lambda name:-scores[name]['mean']);diff=np.array(contributions[order[0]])-np.array(contributions[order[1]])
    return {'candidate_id':order[0],'scores':scores,'per_draw':contributions,'runner_up':order[1],'gap':float(diff.mean()),'paired_gap_mcse':float(diff.std(ddof=1)/np.sqrt(draws)),'seed':seed,'draws':draws}


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);source=root/'sources'/SELF;source.parent.mkdir(parents=True);shutil.copyfile(ROOT/SELF,source)
    particles=bank();cases=[{'case_id':f'case-{i+1}','truth':particles[i*8+((i*3)%8)],'baseline_seed':3100000+i,'candidate_seed_base':3200000+10000*i,'random_seed':3300000+i,'mc_seed':3400000+i} for i in range(4)]
    protocol={'study':'Original synthetic forced stick-slip vibro-impact oscillator, inspired by contact/friction systems, NOT paper replication',
              'sources_motivation':['https://arxiv.org/abs/2502.12288','https://underactuated.mit.edu/contact.html'],
              'frozen_at':now(),'source_sha256':sha(source),'bank':particles,'cases':cases,'profiles':PROFILES,'baseline':BASELINE,'times':TIMES.tolist(),'noise':NOISE,'policies':POLICIES,'fixed_sequence':FIXED,'rounds':3,'maximum_selector_calls':12,'numerical_runtime_s':600,'job_limit_s':120,
              'contact_approximation':'Reboundspeed<1e-6 becomes wallrest with unilateralforce; inward release requires forceexceedingstaticfriction. Half/quarterthreshold/root tolerance audit before acquisition.',
              'numerical_rule':'Uniform prior32particles/equalclassmass. ExactGaussian history likelihood; maximize expected CLASS mutual information MC128common draws; listedorder ties; no tuning.',
              'metrics':'Primary classidentification efficiency: allround0/1/2/3 negativeclasslogscore, classprobability, entropy/Brier and reserved posteriorpredictive positionRMSE. Finalmetrics separately; parameter uncertainty finitebank only. Fourworlds/no noise replicates; descriptive only.'}
    write(root/'protocol.json',protocol);write(root/'environment.json',{'python':sys.version,'numpy':np.__version__,'device':'cpu','started_at':now(),'monotonic_start':time.monotonic(),'invocation':sys.argv,'usage':None})
    write(root/'input-manifest.json',{'artifacts':inventory(root,['input-manifest.json'])});trace(root,'freeze','Freeze originalmodelclassbank, allprofiles/worlds/noise/metrics/budgets before outcomes.',[root/'protocol.json',source])
    prepare_predictions(root)


def check_runtime(root):
    env=read(root/'environment.json')
    # Active numerical stages recorded separately; selector waiting time is excluded.
    total=sum(read(p)['elapsed_s'] for p in root.glob('stage-*-runtime.json'))
    if total>=600:raise RuntimeError('numerical stage budget exhausted')
    return total

def prepare_predictions(root):
    start=time.monotonic();p=read(root/'protocol.json');rows={};events={}
    for name,profile in {'baseline':BASELINE,**PROFILES}.items():
        trajectories=[];eventrows=[]
        for particle in p['bank']:
            y,e=simulate(particle,profile);trajectories.append(y.tolist());eventrows.append(e)
        rows[name]=trajectories;events[name]=eventrows
    refinement=[]
    for name,profile in {'baseline':BASELINE,**PROFILES}.items():
        for particle in p['bank']:
            y,e=simulate(particle,profile);y2,e2=simulate(particle,profile,collapse_speed=2.5e-7,root_tolerance=2.5e-13)
            same=[a['kind'] for a in e]==[a['kind'] for a in e2];common=[abs(a['time']-b['time']) for a,b in zip(e,e2)] if same else []
            refinement.append({'profile':name,'class':particle['class'],'max_position_change':float(np.max(np.abs(y-y2))),'max_matched_event_time_change':max(common,default=0.) if same else None,'event_count_same':len(e)==len(e2),'event_kind_sequence_same':same})
    write(root/'contact-refinement.json',refinement)
    if max(x['max_position_change'] for x in refinement)>1e-5:raise RuntimeError('declared refinement gate failed')
    write(root/'predictions.json',rows);write(root/'preflight-events.json',events)
    # Reserved profiles declared here in source BEFORE outcomes, distinct from candidates.
    heldout={'heldout_a':[.6,-.9,.2,1.7],'heldout_b':[-1.3,.4,1.1,-.7]}
    write(root/'heldout-predictions.json',{name:[simulate(q,profile)[0].tolist() for q in p['bank']] for name,profile in heldout.items()})
    heldref=[]
    for name,profile in heldout.items():
        for particle in p['bank']:
            a,e=simulate(particle,profile);b,e2=simulate(particle,profile,collapse_speed=2.5e-7,root_tolerance=2.5e-13)
            heldref.append({'profile':name,'max_position_change':float(np.max(np.abs(a-b))),'event_count_same':len(e)==len(e2),'event_kind_sequence_same':[a['kind'] for a in e]==[a['kind'] for a in e2],'max_matched_event_time_change':max([abs(a['time']-b['time']) for a,b in zip(e,e2)],default=0.) if [a['kind'] for a in e]==[a['kind'] for a in e2] else None})
    write(root/'heldout-refinement.json',heldref)
    if max(x['max_position_change'] for x in heldref)>1e-5:raise RuntimeError('heldout refinement gate failed')
    for case in p['cases']:
        folder=root/'cases'/case['case_id'];folder.mkdir(parents=True);truthindex=p['bank'].index(case['truth'])
        obs={'candidate_id':'baseline','positions':(np.array(rows['baseline'][truthindex])+np.random.default_rng(case['baseline_seed']).normal(0,NOISE,len(TIMES))).tolist()}
        for policy in POLICIES:write(folder/policy/'history-0.json',[obs])
    write(root/'prepared-state.json',{'artifacts':{str(q.relative_to(root)):sha(q) for q in root.rglob('*') if q.is_file() and q.name in ['predictions.json','heldout-predictions.json','preflight-events.json','contact-refinement.json','heldout-refinement.json','history-0.json']}})
    write(root/'stage-prepare-runtime.json',{'elapsed_s':time.monotonic()-start});freeze_round(root,1)


def context(history,probabilities,round_number,particles=None,weights=None):
    return {'objective':'Identify unknown friction/contact model class and improve position prediction.','round':round_number,'rounds_total':3,
            'known_model':{'mass':1.,'walls_m':[-.25,.25],'initial_position':0.,'initial_velocity':0.,'sliding':'xdot=v;vdot=F(t)−kx−cv−Fk*sign(v)',
                           'sticking':'At velocityzero stick while|F−kx|≤Fs; release when threshold exceeded. Wallrest holds outward force; release requires inward drive aboveFs. Reboundspeed<1e−6 collapses to wallcontact (declared approximation).','impact':'v+=−e(|v−|)*v− at eitherwall.',
                           'classes':CLASSES,'family_definitions':'coulomb:Fs=Fk;static:Fs=1.6Fk;constant:e=e0;speed:e=clip(e0−.10|v−|,.2,.95)',
                           'balanced_parameter_prior':{'friction':[.16,.22],'restitution':[.72,.82],'spring_damping_pairs':[[2.,.14],[2.4,.22]]}},
            'history':[{'candidate_id':h['candidate_id'],'force_profile':BASELINE if h['candidate_id']=='baseline' else PROFILES[h['candidate_id']], 'segment_start_times_s':[0,1,2,3],'sample_times_s':TIMES.tolist(),'positions_m':h['positions']} for h in history],
            'diagnostics':{'particle_posterior':[{'parameters':q,'probability':float(w)} for q,w in zip(particles if particles is not None else bank(),weights if weights is not None else np.full(32,1/32))],'class_posterior':probabilities,'noise_std_m':NOISE,'limitations':'Finitebank assumedmodel/noise; classposterior may be overconfident or misspecification-sensitive. Positions only; noeventlabels.'},
            'candidates':[{'id':name,'force_N':profile,'segment_start_times_s':[0,1,2,3],'sample_times_s':TIMES.tolist(),'measurement_count':16} for name,profile in PROFILES.items()]}

def freeze_round(root,r):
    check_runtime(root);start=time.monotonic();p=read(root/'protocol.json');pred={k:np.array(v) for k,v in read(root/'predictions.json').items()};ledger=[]
    for case in p['cases']:
        for policy in POLICIES:
            route=root/'cases'/case['case_id']/policy;history=read(route/f'history-{r-1}.json');w,prob=posterior(history,pred,p['bank'])
            write(route/f'posterior-{r-1}.json',{'weights':w.tolist(),'class_probabilities':prob})
            if policy=='fresh_llm':
                folder=route/f'selector-{r}';write(folder/'context.json',context(history,prob,r,p['bank'],w));(folder/'instructions.txt').write_text(INSTRUCTIONS)
                ledger.append({'case_id':case['case_id'],'context':str((folder/'context.json').resolve()),'instructions':str((folder/'instructions.txt').resolve())})
            else:
                if policy=='class_information':decision=class_information(w,pred,p['bank'],case['mc_seed']+r*100)
                elif policy=='fixed_balanced':decision={'candidate_id':FIXED[r-1],'rationale':'Frozen balanced diagnostic sequence, not learned choice.'}
                else:decision={'candidate_id':list(PROFILES)[int(np.random.default_rng(case['random_seed']+100*r).integers(len(PROFILES)))],'rationale':'Frozen seeded uniformrandom.'}
                write(route/f'decision-{r}.json',{'policy':policy,**decision})
    write(root/f'round-{r}-contexts.json',ledger)
    write(root/f'round-{r}-frozen.json',{'artifacts':{str(q.relative_to(root)):sha(q) for q in root.rglob('*') if q.is_file() and (q.name in [f'history-{r-1}.json',f'posterior-{r-1}.json',f'decision-{r}.json'] or f'selector-{r}' in q.parts)}})
    write(root/f'stage-freeze-{r}-runtime.json',{'elapsed_s':time.monotonic()-start});trace(root,'decision',f'Freeze round{r} numerical/fixed/random decisions and4actualmodelcontexts before newmeasurements.',[root/f'round-{r}-frozen.json',root/f'round-{r}-contexts.json']);print('CONTEXTS='+str((root/f'round-{r}-contexts.json').resolve()),flush=True)


def check_hashes(root,name):
    for rel,digest in read(root/name)['artifacts'].items():
        if sha(root/rel)!=digest:raise ValueError('artifact mismatch:'+rel)

def acquire(root,r,map_path):
    if (root/f'round-{r}-lock.json').exists():raise ValueError('round alreadystarted, no retries')
    if r>1 and not (root/f'round-{r-1}-complete.json').exists():raise ValueError('previous round incomplete')
    verify(root);check_hashes(root,f'round-{r}-frozen.json');entries=read(map_path);expected={x['case_id'] for x in read(root/f'round-{r}-contexts.json')}
    if len(entries)!=4 or {x['case_id'] for x in entries}!=expected:raise ValueError('decision map mustcontain fouruniqueactualcases')
    for entry in entries:
        d=read(Path(entry['decision']));read(Path(entry['metadata']))
        if set(d)!={'candidate_id','hypothesis','rationale'} or d['candidate_id'] not in PROFILES or any(not isinstance(d[k],str) or not 0<len(d[k])<700 for k in ['hypothesis','rationale']):raise ValueError('invalidactualdecision')
    write(root/f'round-{r}-lock.json',{'started_at':now()});start=time.monotonic();shutil.copyfile(map_path,root/f'round-{r}-decision-map.json')
    for entry in entries:
        route=root/'cases'/entry['case_id']/'fresh_llm';shutil.copyfile(entry['decision'],route/f'decision-{r}.json');shutil.copyfile(entry['metadata'],route/f'metadata-{r}.json')
    p=read(root/'protocol.json');pred=read(root/'predictions.json')
    for case in p['cases']:
        truthindex=p['bank'].index(case['truth'])
        for policy in POLICIES:
            route=root/'cases'/case['case_id']/policy;choice=read(route/f'decision-{r}.json')['candidate_id'];seed=case['candidate_seed_base']+100*r+list(PROFILES).index(choice)
            y=np.array(pred[choice][truthindex])+np.random.default_rng(seed).normal(0,NOISE,len(TIMES));old=read(route/f'history-{r-1}.json');obs={'candidate_id':choice,'positions':y.tolist()}
            write(route/f'new-observations-{r}.json',obs);write(route/f'history-{r}.json',old+[obs]);trace(root,'action',f'{case["case_id"]}/{policy} round{r}: {choice}; sharedcandidate-roundevaluatornoise.',[route/f'decision-{r}.json',route/f'history-{r}.json'])
    write(root/f'stage-acquire-{r}-runtime.json',{'elapsed_s':time.monotonic()-start});write(root/f'round-{r}-complete.json',{'artifacts':{str(q.relative_to(root)):sha(q) for q in root.rglob(f'history-{r}.json')}})
    if r<3:freeze_round(root,r+1)
    else:finish(root)


def finish(root):
    started=time.monotonic();p=read(root/'protocol.json');pred={k:np.array(v) for k,v in read(root/'predictions.json').items()};held=read(root/'heldout-predictions.json');results=[]
    for case in p['cases']:
        truthindex=p['bank'].index(case['truth']);row={'case_id':case['case_id'],'truth':case['truth'],'policies':{}}
        for policy in POLICIES:
            route=root/'cases'/case['case_id']/policy;stages=[]
            for r in range(4):
                w,prob=posterior(read(route/f'history-{r}.json'),pred,p['bank']);errors=[]
                for values in held.values():
                    means=np.array(values);errors.extend((w@means-means[truthindex]).tolist())
                cp=prob[case['truth']['class']];entropy=-sum(v*np.log(v) for v in prob.values() if v>0);brier=sum((v-(label==case['truth']['class']))**2 for label,v in prob.items());stages.append({'round':r,'class_probabilities':prob,'true_class_probability':cp,'class_negative_logscore':float(-np.log(max(cp,1e-300))),'class_entropy':float(entropy),'class_brier_score':float(brier),'heldout_position_rmse_m':float(np.sqrt(np.mean(np.square(errors)))),'map_class':max(prob,key=prob.get)})
            row['policies'][policy]={'stages':stages,'choices':[read(route/f'decision-{r}.json')['candidate_id'] for r in [1,2,3]]}
        results.append(row)
    aggregate={policy:{'heldout_rmse_m':float(np.mean([row['policies'][policy]['stages'][-1]['heldout_position_rmse_m'] for row in results])), 'class_logscore':float(np.mean([row['policies'][policy]['stages'][-1]['class_negative_logscore'] for row in results])), 'correct_map_class':sum(row['policies'][policy]['stages'][-1]['map_class']==row['truth']['class'] for row in results)} for policy in POLICIES}
    curves={policy:[{metric:float(np.mean([row['policies'][policy]['stages'][r][metric] for row in results])) for metric in ['class_negative_logscore','true_class_probability','class_entropy','class_brier_score','heldout_position_rmse_m']} for r in range(4)] for policy in POLICIES}
    write(root/'results.json',{'cases':results,'aggregate':aggregate,'learning_curves':curves,'interpretation':'Fourfixedsyntheticworlds, one noisestream andactualmodelcall/context; descriptiveonly. Not paperreplication/realworld/modelscaling.'})
    lines=['# Stick-slip/impact model-class pilot','','Original synthetic benchmark with repeated impacts, sticking, release and unknown contact/friction class. This pilot uses posterior inference without PINN optimization.','','|Policy|Heldout position RMSE (m)|Mean class log score|Correct MAP / 4|','|---|---:|---:|---:|']
    for policy,v in aggregate.items():lines.append(f'|{policy}|{v["heldout_rmse_m"]:.6f}|{v["class_logscore"]:.4f}|{v["correct_map_class"]}|')
    lines+=['','|Round|Policy|Mean class log score|Mean true-class probability|Heldout RMSE (m)|','|---:|---|---:|---:|---:|']
    for policy,curve in curves.items():
        for r,v in enumerate(curve):lines.append(f'|{r}|{policy}|{v["class_negative_logscore"]:.4f}|{v["true_class_probability"]:.4f}|{v["heldout_position_rmse_m"]:.6f}|')
    lines+=['','Finite prior bank and assumed Gaussian noise; four worlds without noise replication. Twelve actual fresh selector calls receive sparse positions, known physics and class probabilities. Numerical utility scores and hidden event labels stay outside their contexts. True parameters belong to the inference bank, an optimistic closed-bank test. One call per context leaves selector variability unmeasured. Results do not establish general LLM advantage.','', 'Run/verify with frozen sources/experiments/stick_slip_pilot.py verify --campaign DIR.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');trace(root,'evidence','Complete allfourpolicies/threeacquisitions; compare posteriorpredictive/classidentification, not neuraloptimizer behavior.',[root/'results.json',root/'report.md'])
    write(root/'stage-finish-runtime.json',{'elapsed_s':time.monotonic()-started})
    write(root/'manifest.json',{'status':'complete','numerical_runtime_s':check_runtime(root),'artifacts':inventory(root,['manifest.json'])});verify(root);print('COMPLETE='+str(root.resolve()),flush=True)


def verify(root):
    protocol=read(root/'protocol.json')
    if sha(root/'sources'/SELF)!=protocol['source_sha256']:raise ValueError('sourcechanged')
    check_hashes(root,'input-manifest.json')
    if (root/'prepared-state.json').exists():check_hashes(root,'prepared-state.json')
    for r in [1,2,3]:
        for name in [f'round-{r}-frozen.json',f'round-{r}-complete.json']:
            if (root/name).exists():check_hashes(root,name)
    if (root/'manifest.json').exists():
        check_hashes(root,'manifest.json');m=read(root/'manifest.json')
        if set(m['artifacts'])!=set(inventory(root,['manifest.json'])):raise ValueError('inventorychanged')
        if m.get('numerical_runtime_s',0)>600:raise ValueError('runtimecap')
    for row in map(json.loads,(root/'trace.jsonl').read_text().splitlines()):
        for rel,digest in row['artifacts'].items():
            if sha(root/rel)!=digest:raise ValueError('traceartifactchanged:'+rel)
    print('Source/input/stage/artifact hashes verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','acquire','verify','preflight']);parser.add_argument('--out',type=Path);parser.add_argument('--campaign',type=Path);parser.add_argument('--round',type=int,choices=[1,2,3]);parser.add_argument('--decision-map',type=Path);args=parser.parse_args();root=args.out if args.command=='prepare' else args.campaign;existing=root.exists() if root else False
    def timeout(signum,frame):raise TimeoutError('120second numerical command cap')
    signal.signal(signal.SIGALRM,timeout);remaining=600-check_runtime(root) if root and root.exists() and (root/'environment.json').exists() else 600;signal.alarm(max(1,min(120,int(remaining))))
    try:
        if args.command=='prepare':prepare(root.resolve())
        elif args.command=='acquire':acquire(root.resolve(),args.round,args.decision_map)
        elif args.command=='preflight':print(json.dumps(preflight(),indent=2))
        else:verify(root.resolve())
    except Exception as e:
        if root and root.exists() and not isinstance(e,(ValueError,FileExistsError)):
            (root/'failure-stack.txt').write_text(traceback.format_exc());write(root/'failure.json',{'status':'failed','error':str(e),'aggregate':'unavailable_no_subset_ranking'})
        raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
