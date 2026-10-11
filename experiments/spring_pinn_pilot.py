"""Bounded synthetic lecturer PINN pilot with frozen inputs and artifact checks.

Run: .venv/bin/python experiments/spring_pinn_pilot.py --out data/research/NEW
     --notebook /private/tmp/lecturer-1-pinn.ipynb
Verify: .venv/bin/python experiments/spring_pinn_pilot.py --verify CAMPAIGN
Outputs are local development artifacts; never stage them.
"""
import argparse
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
from torch import nn


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def trace(root, kind, summary, artifacts):
    row = {'timestamp_utc': now(), 'kind': kind, 'summary': summary,
           'artifacts_sha256': {str(Path(p).relative_to(root)): sha(p) for p in artifacts},
           'usage': {'source': 'collaboration API does not expose usage or model identity',
                     'model_id': None, 'input_tokens': None, 'output_tokens': None,
                     'reasoning_tokens': None, 'cached_input_tokens': None}}
    with (root / 'thoughts.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')


def validate(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = manifest['artifacts_sha256']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    if actual != set(expected):
        raise ValueError('artifact inventory mismatch')
    for name, digest in expected.items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or sha(p) != digest:
            raise ValueError(f'artifact hash mismatch: {name}')
    for row in map(json.loads, (root / 'thoughts.jsonl').read_text().splitlines()):
        for name, digest in row['artifacts_sha256'].items():
            if expected.get(name) != digest:
                raise ValueError(f'trace link mismatch: {name}')
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, digest in protocol['frozen_sha256'].items():
        if expected.get(name) != digest:
            raise ValueError(f'input/source hash mismatch: {name}')
    return manifest


def solution(t):
    omega = torch.sqrt(torch.tensor(246.0))
    return torch.exp(-2*t) * (1.5*torch.cos(omega*t) + (8/omega)*torch.sin(omega*t))


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Sequential(nn.Linear(1,32),nn.Tanh()),
                                 nn.Sequential(nn.Linear(32,64),nn.Tanh()),
                                 nn.Sequential(nn.Linear(64,32),nn.Tanh()),nn.Linear(32,1))

    def forward(self,t):
        return self.net(2*t-1)


def residual(model,t,b):
    u = model(t)
    du = torch.autograd.grad(u,t,torch.ones_like(u),create_graph=True)[0]
    ddu = torch.autograd.grad(du,t,torch.ones_like(du),create_graph=True)[0]
    return ddu+b*du+250*u


def worker(root, seed, density):
    protocol = json.loads((root/'protocol.json').read_text())
    for name,digest in protocol['frozen_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError(f'frozen input modified: {name}')
    job = root/f'{density}-seed-{seed}'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = Network()
    b = nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam(list(model.parameters())+[b],lr=5e-4,betas=(.98,.9995),amsgrad=True)
    data = json.loads((root/'observations.json').read_text())
    indices = list(range(150)) if density=='dense' else data['sparse_indices']
    t = torch.tensor(data['times'])[indices].reshape(-1,1)
    y = torch.tensor(data['observed'])[indices].reshape(-1,1)
    physics = torch.linspace(0,1,50).reshape(-1,1).requires_grad_(True)
    start = time.monotonic()
    steps = 0
    status = 'complete'
    with (job/'training.jsonl').open('x') as log:
        for step in range(protocol['target_steps']):
            opt.zero_grad(set_to_none=True)
            physics.grad = None
            ode = residual(model,physics,b).square().mean()
            obs = (model(t)-y).square().mean()
            loss = ode+1500*obs
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            opt.step()
            steps = step+1
            if steps==1 or steps%1000==0 or steps==protocol['target_steps']:
                row = {'step':steps,'elapsed_s':time.monotonic()-start,'damping':b.item(),
                       'loss':loss.item(),'ode_loss':ode.item(),'observation_loss':obs.item()}
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
            if time.monotonic()-start >= protocol['training_deadline_s']:
                status = 'training_budget_exhausted'
                break
    grid = torch.linspace(0,1,1001).reshape(-1,1).requires_grad_(True)
    r = residual(model,grid,b).detach()
    with torch.no_grad():
        pred = model(grid)
        error = pred-solution(grid)
        grid300 = torch.linspace(0,1,300).reshape(-1,1)
        upstream_rmse = torch.sqrt(torch.trapezoid((model(grid300)-solution(grid300)).flatten().square(),grid300.flatten()))
    torch.save({'network_state_dict':model.state_dict(),'damping_parameter':b.detach(),
                'optimizer_state_dict':opt.state_dict(),'completed_steps':steps},job/'checkpoint.pt')
    write(job/'curves.json',{'times':grid.detach().flatten().tolist(),'prediction':pred.flatten().tolist(),
                            'reference':solution(grid).detach().flatten().tolist(),'ode_residual':r.flatten().tolist()})
    result = {'status':status,'seed':seed,'density':density,'completed_steps':steps,
              'target_steps':protocol['target_steps'],'learned_damping':b.item(),
              'damping_absolute_error':abs(b.item()-4),'trajectory_rmse':error.square().mean().sqrt().item(),
              'upstream_integrated_rmse_300':upstream_rmse.item(),'dense_grid_residual_rmse':r.square().mean().sqrt().item(),
              'elapsed_s':time.monotonic()-start,'completed_at_utc':now(),
              'learned_network_weights':'checkpoint.pt:network_state_dict',
              'learned_physical_parameter':'checkpoint.pt:damping_parameter'}
    write(job/'result.json',result)
    print(json.dumps(result),flush=True)


def run_job(root, job, command, timeout):
    """Persist exceptions/stacks and partial files even when a worker fails."""
    write(job/'invocation.json',{'argv':command,'started_at_utc':now(),'timeout_s':timeout})
    started = time.monotonic()
    with (job/'stdout.log').open('x') as out, (job/'stderr.log').open('x') as err:
        try:
            completed = subprocess.run(command,stdout=out,stderr=err,timeout=timeout,check=True)
            status = {'returncode':completed.returncode,'status':'worker_finished'}
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
            stack = traceback.format_exc()
            (job/'controller-stack.txt').write_text(stack)
            status = {'status':'worker_failed','error_type':type(exc).__name__,'error':str(exc)}
    status.update({'elapsed_s':time.monotonic()-started,'completed_at_utc':now()})
    write(job/'process.json',status)
    return status


def campaign(out, notebook):
    out.mkdir(parents=True,exist_ok=False)
    sources = out/'sources'; sources.mkdir()
    shutil.copyfile(__file__,sources/'spring_pinn_pilot.py')
    shutil.copyfile(notebook,sources/'upstream.ipynb')
    torch.manual_seed(1)
    times = torch.linspace(0,1,150).reshape(-1,1)
    observed = solution(times)+.05*torch.randn_like(times)
    write(out/'observations.json',{'times':times.flatten().tolist(),'observed':observed.flatten().tolist(),
                                  'sparse_indices':np.rint(np.linspace(0,149,30)).astype(int).tolist()})
    frozen = {str(p.relative_to(out)):sha(p) for p in [sources/'spring_pinn_pilot.py',sources/'upstream.ipynb',out/'observations.json']}
    protocol = {'study':'synthetic spring damping identification development pilot',
                'target_steps':80000,'training_deadline_s':110,'job_timeout_s':120,'campaign_compute_budget_s':600,
                'maximum_jobs':6,'jobs':[{'seed':s,'density':d} for s in [1,2,3] for d in ['dense','sparse']],
                'physics':{'m':1,'k':250,'true_b':4,'y0':1.5,'v0':5,'horizon':1},
                'noise_std':.05,'observation_data_seed':1,'dense_observations':150,'sparse_observations':30,
                'network':[32,64,32],'activation':'Tanh','initial_b':0,'collocation_points':50,
                'optimizer':{'name':'Adam','lr':.0005,'betas':[.98,.9995],'amsgrad':True},
                'observation_loss_weight':1500,'dtype':'float32','device':'cpu','threads':1,
                'decision_rule':'No tuning. Preserve incomplete jobs. Dense seed1 is upstream replication only if all 80000 steps finish.',
                'frozen_sha256':frozen,'frozen_at_utc':now()}
    write(out/'protocol.json',protocol)
    write(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                                  'scipy':scipy.__version__,'device':'cpu','platform':platform.system(),
                                  'invocation':sys.argv,'started_at_utc':now()})
    trace(out,'hypothesis','Fewer observations may impair damping recovery; compare paired initializations on one frozen noisy dataset without tuning.',[out/'protocol.json',out/'observations.json',*sources.iterdir()])
    spent = 0.0
    for spec in protocol['jobs']:
        job = out/f"{spec['density']}-seed-{spec['seed']}"; job.mkdir()
        if spent>=600:
            write(job/'process.json',{'status':'campaign_budget_exhausted','completed_at_utc':now()})
        else:
            command = [sys.executable,str(sources/'spring_pinn_pilot.py'),'--worker',str(out),'--seed',str(spec['seed']),'--density',spec['density']]
            result = run_job(out,job,command,min(120,600-spent)); spent += result['elapsed_s']
        artifacts = list(job.iterdir())
        trace(out,'evidence',f"Retained {job.name} outputs and failure information; no protocol changes.",artifacts)
    results = []
    for spec in protocol['jobs']:
        job = out/f"{spec['density']}-seed-{spec['seed']}"
        result_path = job/'result.json'
        results.append(json.loads(result_path.read_text()) if result_path.exists() else {**spec, **json.loads((job/'process.json').read_text())})
    lines = ['# Lecturer PINN learning pilot','',
             'Synthetic development data only. Frozen protocol, three paired initialization seeds, no tuning. All seeds share one measurement-noise realization: this measures initialization variability, not independent data/noise replication. Partial training is not an 80k-step reproduction.', '',
             '|Data|Seed|Status|Steps|Damping|Absolute error|Trajectory RMSE|Residual RMSE|Seconds|',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        if 'learned_damping' not in r:
            lines.append(f"|{r['density']}|{r['seed']}|{r['status']}|unavailable|—|—|—|—|{r.get('elapsed_s', 0):.2f}|")
            continue
        lines.append(f"|{r['density']}|{r['seed']}|{r['status']}|{r['completed_steps']}|{r['learned_damping']:.6f}|{r['damping_absolute_error']:.6f}|{r['trajectory_rmse']:.6f}|{r['dense_grid_residual_rmse']:.6f}|{r['elapsed_s']:.2f}|")
    lines += ['', 'Neural weights and learned physical damping are stored separately in each checkpoint. Trajectory RMSE uses 1001 uniform points; upstream-integrated RMSE uses 300 points and trapezoidal integration. Dense residual is evaluated on 1001 points using learned damping. Failure logs/stacks and partial progress are retained.', '',
              'Learning guide: the network maps time to displacement. Autograd computes two time derivatives. The objective adds squared ODE residual to 1500 times observation error; mass and spring constant are fixed, damping is learned from zero. Initial displacement/velocity generate data but are not separately enforced as boundary losses.', '',
              'The first two hybrid-systems sections (updated local notes PDF pages4–28, numbered slides3–25) introduce continuous flows, discrete jumps, the (C,F,D,G) model, hybrid time (t,j), solution properties, and Zeno accumulation. A bouncing ball flows under gravity and jumps at impact via velocity restitution. A single smooth displacement network cannot directly represent discontinuous velocity; a future hybrid pilot should treat flight segments and resets explicitly. This campaign runs no hybrid experiment.', '',
              'The public upstream directory provides 1_pinn.ipynb; its link to 2_pinn.ipynb is missing. Meeting questions: how to choose the loss weight; when damping recovery is reliable; whether impact times are observed or inferred; and how to model continuation after Zeno accumulation.', '',
              'Run: `.venv/bin/python experiments/spring_pinn_pilot.py --out data/research/NEW --notebook /private/tmp/lecturer-1-pinn.ipynb`. Verify: `.venv/bin/python experiments/spring_pinn_pilot.py --verify CAMPAIGN`.', '',
              f'Total bounded subprocess compute: {spent:.2f}s. Interpret only the retained synthetic runs; three seeds do not establish general robustness.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    trace(out,'next_step','Review paired damping and trajectory errors, training completion and residuals before planning a separate hybrid pilot.',[out/'report.md'])
    counts = {status:sum(r['status']==status for r in results) for status in sorted({r['status'] for r in results})}
    status = 'completed' if counts.get('complete') == 6 else 'completed_with_failures_or_incomplete'
    manifest = {'schema_version':1,'status':status,'job_status_counts':counts,'completed_at_utc':now(),'compute_elapsed_s':spent,
                'artifacts_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    write(out/'manifest.json',manifest)
    validate(out)
    print(f'Validated campaign: {out}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path); parser.add_argument('--notebook',type=Path)
    parser.add_argument('--verify',type=Path); parser.add_argument('--worker',type=Path)
    parser.add_argument('--seed',type=int); parser.add_argument('--density',choices=['dense','sparse'])
    args = parser.parse_args()
    if args.verify:
        validate(args.verify); print('Artifact and trace validation passed')
    elif args.worker:
        worker(args.worker,args.seed,args.density)
    elif args.out and args.notebook:
        campaign(args.out.resolve(),args.notebook)
    else:
        parser.error('provide --out and --notebook, or --verify')


if __name__=='__main__':
    main()
