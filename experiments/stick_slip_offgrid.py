"""Frozen off-grid stick-slip/impact parameter-mismatch extension, not mechanism discovery.
prepare --out NEW; acquire --campaign DIR --round N --decision-map JSON;
verify --campaign DIR. Execute the frozen sources/experiments/stick_slip_offgrid.py.
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
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
SELF='experiments/stick_slip_offgrid.py'
CLASSES=['coulomb_constant','coulomb_speed','static_constant','static_speed']
POLICIES=['fresh_llm','class_information','fixed_balanced','uniform_random']
PROFILES={'breakaway':[.18,.26,.34,.42],'gentle_reversal':[.30,-.30,.30,-.30],
          'impact_reversal':[1.2,-1.2,1.2,-1.2],'high_speed':[2.,-2.,2.,-2.],
          'asymmetric':[.5,1.5,-.4,-1.5],'release_rest':[1.,0.,-1.,0.],
          'threshold_steps':[.20,.28,-.20,-.28],'short_push':[1.6,0.,0.,-.3],
          'rising_reversal':[.4,.8,-1.2,-1.6],'alternating_strength':[.35,-1.4,.35,-1.4]}
RESERVED={'heldout_a':[.6,-.9,.2,1.7],'heldout_b':[-1.3,.4,1.1,-.7],'threshold_probe':[.24,.24,-.24,-.24]}
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
    # Preserve the initial state and tiny departures without cancellation of
    # equilibrium-sized terms (especially when a segment starts on a wall).
    displacement=b*(np.expm1(-a*t)-2*exp*np.sin(w*t/2)**2)+exp*d*si
    return x+displacement,exp*(v*co+(-a*v-p['spring']*b)/w*si)

def simulate(p,profile,times=TIMES,collapse_speed=1e-6,root_tolerance=1e-12):
    """Exact damped linear slip segments with analytic velocity-zero time and monotone wall root.
    Stick releases at force discontinuities. No penetration clipping or hidden chatter fix.
    """
    fs=p['friction']*(1.6 if p['class'].startswith('static') else 1.)
    x=v=0.;segments=[];events=[];count=0
    for interval,force in enumerate(profile):
        t=float(interval);end=float(interval+1)
        while t<end:
            count+=1
            if count>300:raise RuntimeError('event/chatter cap exceeded')
            net=force-p['spring']*x
            if v==0.:
                atwall=abs(x)>=.25
                if atwall and x*net>=-abs(x)*fs:
                    segments.append((t,end,x,0.,force,0));events.append({'time':t,'kind':'wall_rest','position':x});t=end;continue
                if abs(net)<=fs+1e-12:
                    segments.append((t,end,x,0.,force,0));events.append({'time':t,'kind':'stick','position':x});t=end;continue
                direction=float(np.sign(net));events.append({'time':t,'kind':'release','position':x})
            else:direction=float(np.sign(v))
            duration=end-t;alpha=p['damping']/2;omega=np.sqrt(p['spring']-alpha*alpha);eq=(force-direction*p['friction'])/p['spring']
            angle=float(np.arctan2(-v,(-alpha*v-p['spring']*(x-eq))/omega)%np.pi)
            # At rest the zero at the segment origin is departure, not a stop.
            # A nonzero incoming velocity can have a genuinely very near stop.
            if v==0.:angle=np.pi
            stop=angle/omega;dt=min(duration,stop);kind='velocity' if stop<=duration else None
            endpoint=float(slip(dt,x,v,force,p,direction)[0])
            # Velocity has constant sign up to stop. Only the wall ahead can
            # be hit; the wall just departed must never supply the t=0 root.
            wall=.25 if direction>0 else -.25
            if direction*(endpoint-wall)>=0:
                bracket_end=dt
                wall_residual=lambda z:float(slip(z,x,v,force,p,direction)[0])-wall
                dt=brentq(wall_residual,0.,bracket_end,xtol=root_tolerance)
                # An absolute root tolerance may round a real near-origin hit
                # to zero. Resolve that root before deciding time cannot advance.
                if t+dt<=t and x!=wall:
                    dt=brentq(wall_residual,0.,bracket_end,xtol=np.nextafter(0.,1.))
                kind='right' if wall>0 else 'left'
            # A stop arbitrarily near a force boundary leaves a valid short
            # remainder. Reject only failure to advance floating-point time.
            next_t=end if dt==duration else t+dt
            if next_t<=t:raise RuntimeError('nonadvancing event/chatter')
            segments.append((t,next_t,x,v,force,direction));xnew,vnew=slip(dt,x,v,force,p,direction);x=float(xnew);v=float(vnew);t=next_t
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
    tests=root/'sources/tests';tests.mkdir();shutil.copyfile(ROOT/'tests/test_stick_slip_offgrid.py',tests/'test_stick_slip_offgrid.py')
    particles=bank();cases=offgrid_cases()
    protocol={'study':'Frozen off-grid parameter/model mismatch robustness extension; four declared mechanisms, not novel mechanism discovery or named model superiority',
              'sources_motivation':['https://arxiv.org/abs/2502.12288','https://underactuated.mit.edu/contact.html'],
              'frozen_at':now(),'source_sha256':sha(source),'bank':particles,'cases':cases,'profiles':PROFILES,'baseline':BASELINE,'times':TIMES.tolist(),'noise':NOISE,'policies':POLICIES,'fixed_sequence':FIXED,'rounds':3,'maximum_selector_calls':24,'reserved_profiles':RESERVED,'map_tie_rule':'Absolute probability tolerance 1e-12; first declared class for MAP, ties separately flagged', 'mismatch_diagnostic':'Minimum bank SSE divided by observed position count; descriptive, not a calibrated significance test','numerical_runtime_s':600,'job_limit_s':120,
              'contact_approximation':'Reboundspeed<1e-6 becomes wallrest with unilateralforce; inward release requires forceexceedingstaticfriction. Half/quarterthreshold/root tolerance audit before acquisition.',
              'numerical_rule':'Uniform prior32particles/equalclassmass. ExactGaussian history likelihood; maximize expected CLASS mutual information MC128common draws; listedorder ties; no tuning.',
              'metrics':'All rounds 0/1/2/3: stable log-space true-class negative log score, true-class probability, MAP correctness and tolerance tie flags, confident wrong (max class probability > .95 and MAP wrong), entropy/Brier, posterior predictive reserved position RMSE per profile and aggregate, minimum bank SSE per measured position. Eight off-grid cases: four classes times two independent noise replicates; descriptive only. Aggregate RMSE is mean of case-level RMSE; two replicates share class truth, independent noise seed streams.'}
    write(root/'protocol.json',protocol);write(root/'environment.json',{'python':sys.version,'numpy':np.__version__,'device':'cpu','started_at':now(),'monotonic_start':time.monotonic(),'invocation':sys.argv,'usage':None})
    write(root/'input-manifest.json',{'artifacts':inventory(root,['input-manifest.json'])});trace(root,'freeze','Freeze off-grid truths, inference bank, all candidate/reserved profiles, seeds, metrics and budgets before outcomes.',[root/'protocol.json',source])
    prepare_predictions(root)


def check_runtime(root):
    env=read(root/'environment.json')
    # Active numerical stages recorded separately; selector waiting time is excluded.
    total=sum(read(p)['elapsed_s'] for p in root.glob('stage-*-runtime.json'))
    if total>=600:raise RuntimeError('numerical stage budget exhausted')
    return total

def offgrid_cases():
    return [{'case_id':f'case-{2*i+j+1}', 'replicate':j+1,
             'truth':{'class':label,'friction':.19,'restitution':.77,'spring':2.2,'damping':.18},
             'baseline_seed':4100000+100*i+j,'candidate_seed_base':4200000+10000*(2*i+j),
             'random_seed':4300000+100*i+j,'mc_seed':4400000+100*i+j}
            for i,label in enumerate(CLASSES) for j in range(2)]

def reference(p,profile,times):
    x=v=0.;pieces=[];events=[];fs=p['friction']*(1.6 if p['class'].startswith('static') else 1.)
    for j,force in enumerate(profile):
        t=float(j);end=float(j+1);count=0
        while t<end:
            count+=1
            if count>300:raise RuntimeError('Independent reference event cap')
            drive=force-p['spring']*x
            if v==0.:
                wall=abs(x)>=.25
                if (wall and x*drive>=-abs(x)*fs) or abs(drive)<=fs+1e-12:
                    pieces.append((t,end,None,x));t=end;continue
                direction=np.sign(drive)
            else:direction=np.sign(v)
            def rhs(_,y):return [y[1],force-p['spring']*y[0]-p['damping']*y[1]-direction*p['friction']]
            def velocity(_,y):return direction*y[1]
            def right(_,y):return .25-y[0] if direction>0 else 1.
            def left(_,y):return y[0]+.25 if direction<0 else 1.
            for f in [velocity,right,left]:f.terminal=True;f.direction=-1
            result=solve_ivp(rhs,[t,end],[x,v],events=[velocity,right,left],dense_output=True,rtol=1e-10,atol=1e-12,max_step=.01)
            assert result.success
            newt=float(result.t[-1])
            if newt<=t:raise RuntimeError('Nonadvancing independent event')
            pieces.append((t,newt,result.sol,None));x,v=map(float,result.y[:,-1]);t=newt
            hit=next((i for i,ev in enumerate(result.t_events) if len(ev)),None)
            if hit is None:continue
            if hit==0:v=0.
            else:
                events.append((t,'right' if hit==1 else 'left'))
                e=p['restitution']-(.10*abs(v) if p['class'].endswith('speed') else 0.)
                v=-np.clip(e,.2,.95)*v;x=.25 if hit==1 else -.25
                if abs(v)<1e-6:v=0.
    positions=[]
    for t in times:
        segment=next(s for s in pieces if s[0]-1e-11<=t<=s[1]+1e-11)
        positions.append(segment[3] if segment[2] is None else float(segment[2](t)[0]))
    return np.array(positions),events


def prepare_predictions(root):
    from scipy.integrate import solve_ivp
    start=time.monotonic();p=read(root/'protocol.json');allprofiles={'baseline':BASELINE,**PROFILES,**RESERVED}
    rows={name:[simulate(q,profile)[0].tolist() for q in p['bank']] for name,profile in allprofiles.items()}
    truths={case['case_id']:{name:simulate(case['truth'],profile)[0].tolist() for name,profile in allprofiles.items()} for case in p['cases']}
    truth_events={case['case_id']:{name:simulate(case['truth'],profile)[1] for name,profile in allprofiles.items()} for case in p['cases']}
    refinement=[];references=[]
    unique_truths=[c['truth'] for c in p['cases'][::2]]
    for kind,parameters in [('bank',p['bank']),('offgrid_truth',unique_truths)]:
        for index,q in enumerate(parameters):
            for name,profile in allprofiles.items():
                y,e=simulate(q,profile);y2,e2=simulate(q,profile,collapse_speed=2.5e-7,root_tolerance=2.5e-13)
                same=[a['kind'] for a in e]==[a['kind'] for a in e2]
                refinement.append({'population':kind,'index':index,'profile':name,'max_position_change':float(np.max(abs(y-y2))),'event_kind_sequence_same':same})
                if kind=='offgrid_truth' or index in [0,7,8,15,16,23,24,31]:
                    yr,er=reference(q,profile,TIMES);ea=[(a['time'],a['kind']) for a in e if a['kind'] in ['left','right']]
                    match=[a[1] for a in ea]==[a[1] for a in er]
                    delta=max((abs(a[0]-b[0]) for a,b in zip(ea,er)),default=0.)
                    references.append({'population':kind,'index':index,'profile':name,'position_max_error_m':float(np.max(abs(y-yr))),'impact_sequence_identical':match,'impact_time_max_error_s':delta})
    write(root/'contact-refinement.json',refinement);write(root/'reference-verification.json',references)
    if any(x['max_position_change']>1e-5 or not x['event_kind_sequence_same'] for x in refinement):raise RuntimeError('refinement gate failed')
    if any(x['position_max_error_m']>1e-7 or x['impact_time_max_error_s']>1e-7 or not x['impact_sequence_identical'] for x in references):raise RuntimeError('reference gate failed')
    write(root/'predictions.json',{k:v for k,v in rows.items() if k not in RESERVED})
    write(root/'heldout-predictions.json',{k:rows[k] for k in RESERVED})
    write(root/'truth-predictions.json',truths);write(root/'truth-events.json',truth_events)
    for case in p['cases']:
        folder=root/'cases'/case['case_id'];folder.mkdir(parents=True)
        obs={'candidate_id':'baseline','positions':(np.array(truths[case['case_id']]['baseline'])+np.random.default_rng(case['baseline_seed']).normal(0,NOISE,len(TIMES))).tolist()}
        for policy in POLICIES:write(folder/policy/'history-0.json',[obs])
    write(root/'prepared-state.json',{'artifacts':{str(q.relative_to(root)):sha(q) for q in root.rglob('*') if q.is_file() and q.name in ['predictions.json','heldout-predictions.json','truth-predictions.json','truth-events.json','contact-refinement.json','reference-verification.json','history-0.json']}})
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
    write(root/f'stage-freeze-{r}-runtime.json',{'elapsed_s':time.monotonic()-start});trace(root,'decision',f'Freeze round{r} numerical/fixed/random decisions and8actualmodelcontexts before newmeasurements.',[root/f'round-{r}-frozen.json',root/f'round-{r}-contexts.json']);print('CONTEXTS='+str((root/f'round-{r}-contexts.json').resolve()),flush=True)


def check_hashes(root,name):
    for rel,digest in read(root/name)['artifacts'].items():
        if sha(root/rel)!=digest:raise ValueError('artifact mismatch:'+rel)

def acquire(root,r,map_path):
    if (root/f'round-{r}-lock.json').exists():raise ValueError('round alreadystarted, no retries')
    if r>1 and not (root/f'round-{r-1}-complete.json').exists():raise ValueError('previous round incomplete')
    verify(root);check_hashes(root,f'round-{r}-frozen.json');entries=read(map_path);expected={x['case_id'] for x in read(root/f'round-{r}-contexts.json')}
    if len(entries)!=8 or {x['case_id'] for x in entries}!=expected:raise ValueError('decision map mustcontain eightuniqueactualcases')
    for entry in entries:
        d=read(Path(entry['decision']));read(Path(entry['metadata']))
        if set(d)!={'candidate_id','hypothesis','rationale'} or d['candidate_id'] not in PROFILES or any(not isinstance(d[k],str) or not 0<len(d[k])<700 for k in ['hypothesis','rationale']):raise ValueError('invalidactualdecision')
    write(root/f'round-{r}-lock.json',{'started_at':now()});start=time.monotonic();shutil.copyfile(map_path,root/f'round-{r}-decision-map.json')
    for entry in entries:
        route=root/'cases'/entry['case_id']/'fresh_llm';shutil.copyfile(entry['decision'],route/f'decision-{r}.json');shutil.copyfile(entry['metadata'],route/f'metadata-{r}.json')
    p=read(root/'protocol.json');truths=read(root/'truth-predictions.json')
    for case in p['cases']:
        for policy in POLICIES:
            route=root/'cases'/case['case_id']/policy;choice=read(route/f'decision-{r}.json')['candidate_id'];seed=case['candidate_seed_base']+100*r+list(PROFILES).index(choice)
            y=np.array(truths[case['case_id']][choice])+np.random.default_rng(seed).normal(0,NOISE,len(TIMES));old=read(route/f'history-{r-1}.json');obs={'candidate_id':choice,'positions':y.tolist()}
            write(route/f'new-observations-{r}.json',obs);write(route/f'history-{r}.json',old+[obs]);trace(root,'action',f'{case["case_id"]}/{policy} round{r}: {choice}; sharedcandidate-roundevaluatornoise.',[route/f'decision-{r}.json',route/f'history-{r}.json'])
    write(root/f'stage-acquire-{r}-runtime.json',{'elapsed_s':time.monotonic()-start});write(root/f'round-{r}-complete.json',{'artifacts':{str(q.relative_to(root)):sha(q) for q in root.rglob(f'history-{r}.json')}})
    if r<3:freeze_round(root,r+1)
    else:finish(root)


def stage_metrics(history,pred,held,truth,particles):
    w,prob=posterior(history,pred,particles);errors=[];by_profile={}
    for name,values in held.items():
        error=w@np.array(values)-np.array(truth[name]);errors.extend(error.tolist());by_profile[name]=float(np.sqrt(np.mean(error**2)))
    label=truth['class'];cp=prob[label];top=max(prob.values());ties=[k for k,v in prob.items() if abs(v-top)<=1e-12];mapclass=max(prob,key=prob.get)
    sse=sum(np.sum((pred[h['candidate_id']]-np.array(h['positions']))**2,axis=1) for h in history)
    # Compute class log probability in log space to retain extreme misspecification scores.
    logw=-sse/(2*NOISE**2);logw-=logsumexp(logw)
    logcp=logsumexp(logw[[i for i,q in enumerate(particles) if q['class']==label]])
    return {'class_probabilities':prob,'true_class_probability':cp,'class_negative_logscore':float(-logcp),
            'class_entropy':float(-sum(v*np.log(v) for v in prob.values() if v>0)),
            'class_brier_score':float(sum((v-(k==label))**2 for k,v in prob.items())),
            'heldout_position_rmse_m':float(np.sqrt(np.mean(np.square(errors)))), 'reserved_rmse_m':by_profile,
            'map_class':mapclass,'map_correct':mapclass==label,'map_tie':len(ties)>1,'map_tied_classes':ties,
            'confident_wrong':top>.95 and mapclass!=label,'minimum_bank_sse_per_measurement':float(min(sse)/(len(history)*len(TIMES)))}

def finish(root):
    started=time.monotonic();p=read(root/'protocol.json');pred={k:np.array(v) for k,v in read(root/'predictions.json').items()};held=read(root/'heldout-predictions.json');truths=read(root/'truth-predictions.json');results=[]
    for case in p['cases']:
        row={'case_id':case['case_id'],'truth':case['truth'],'replicate':case['replicate'],'policies':{}}
        for policy in POLICIES:
            route=root/'cases'/case['case_id']/policy
            stages=[{'round':r,**stage_metrics(read(route/f'history-{r}.json'),pred,held,{**truths[case['case_id']],'class':case['truth']['class']},p['bank'])} for r in range(4)]
            row['policies'][policy]={'stages':stages,'choices':[read(route/f'decision-{r}.json')['candidate_id'] for r in [1,2,3]]}
        results.append(row)
    metrics=['class_negative_logscore','true_class_probability','class_entropy','class_brier_score','heldout_position_rmse_m','minimum_bank_sse_per_measurement']
    curves={policy:[{**{m:float(np.mean([row['policies'][policy]['stages'][r][m] for row in results])) for m in metrics},
                         **{m:sum(row['policies'][policy]['stages'][r][m] for row in results) for m in ['map_correct','map_tie','confident_wrong']},
                         'reserved_rmse_m':{name:float(np.mean([row['policies'][policy]['stages'][r]['reserved_rmse_m'][name] for row in results])) for name in RESERVED}} for r in range(4)] for policy in POLICIES}
    write(root/'results.json',{'cases':results,'aggregate':{k:v[-1] for k,v in curves.items()},'learning_curves':curves,'interpretation':'Eight synthetic off-grid cases, two noise replicates per class. Parameter/model mismatch robustness within declared mechanisms, not novel mechanism discovery or named model superiority. Descriptive finite-bank probabilities and SSE diagnostic, no calibrated significance test.'})
    trace(root,'evidence','Completed four policies, three rounds and 24 actual fresh selector calls.',[root/'results.json'])
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
