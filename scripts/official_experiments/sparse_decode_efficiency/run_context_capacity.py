"""Run existing capacity sweeps, then canonical request probes at each verified maximum."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from .sweep_decode_capacity import load_campaign_config, resolve_lane
else:
    from sweep_decode_capacity import load_campaign_config, resolve_lane


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    cfg = load_campaign_config(args.config)
    repo = args.repo.resolve()
    root = Path(cfg['output_root'])
    package = repo/'scripts/official_experiments/sparse_decode_efficiency'
    model_id = cfg['model_id']
    model = cfg['models'][model_id]
    lanes = cfg['lanes']
    for lane in lanes:
        engine, _, external = resolve_lane(cfg, lane)
        if engine not in ('sparseengine', 'vllm'):
            raise ValueError(
                f'{lane}: request probes do not support engine={engine!r}; '
                'use sweep_decode_capacity.py for decode-only measurements.'
            )
        if external:
            if {'CUDA_VISIBLE_DEVICES', 'PYTHONPATH'} & external.get('environment', {}).keys():
                raise ValueError('External environment cannot override CUDA_VISIBLE_DEVICES or PYTHONPATH')
            if (external.get('environment_kind') == 'venv'
                    and not (Path(external['env'])/'pyvenv.cfg').is_file()):
                raise ValueError(f"Configured venv lacks pyvenv.cfg: {external['env']}")
    root.mkdir(parents=True, exist_ok=True)
    if (root/'status.tsv').exists():
        raise FileExistsError('Use a new output root; existing attempts must remain intact')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus,
           'PYTHONPATH': f'{repo}/src:{repo}', 'TOKENIZERS_PARALLELISM': 'false',
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    for key, path in {'TMPDIR': Path(cfg['scratch_root']),
                      'TRITON_CACHE_DIR': root/'cache/triton',
                      'VLLM_CACHE_ROOT': root/'cache/vllm',
                      'TORCHINDUCTOR_CACHE_DIR': root/'cache/inductor',
                      'CUDA_CACHE_PATH': root/'cache/cuda',
                      'XDG_CACHE_HOME': root/'cache/xdg'}.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    active = guard = None
    errors = []
    def status(stage, state, **extra):
        row = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), stage=stage, status=state, **extra)
        print(json.dumps(row), flush=True)
        with (root/'status.tsv').open('a') as f:
            f.write(json.dumps(row)+'\n')
    def stop_group(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(f'Failed to stop campaign process group {process.pid}') from error
    def interrupted(signum, frame):
        raise InterruptedError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def idle():
        for _ in range(60):
            pids = subprocess.check_output(['nvidia-smi','-i',args.gpus,
                '--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            if not pids:
                return
            time.sleep(5)
        raise RuntimeError('Selected GPU has computing processes after bounded wait')
    def ensure_guard(ready):
        if guard is None or guard.poll() is not None:
            raise RuntimeError('Request GPU reservation exited; inspect request_guard.log')
        if ready.with_suffix('.contention.json').exists():
            raise RuntimeError('External GPU contention invalidates the request run')
    def run(command, log, timeout, *, guard_ready=None, run_env=None):
        nonlocal active
        if guard_ready is not None:
            ensure_guard(guard_ready)
        write(log.with_suffix('.command.json'), command)
        with log.open('x') as f:
            active = subprocess.Popen(command, cwd=repo, env=env if run_env is None else run_env,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + timeout
                while True:
                    if guard_ready is not None:
                        ensure_guard(guard_ready)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        code = active.wait(timeout=min(5, remaining))
                    except subprocess.TimeoutExpired:
                        continue
                    if guard_ready is not None:
                        ensure_guard(guard_ready)
                    return code
            finally:
                stop_group(active)
                active = None
    try:
        write(root/'campaign.json', cfg)
        write(root/'queue_status.json', dict(status='running'))
        write(root/'identity.json', dict(repo=str(repo), gpus=args.gpus,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
            gpu_inventory=subprocess.check_output(['nvidia-smi','-i',args.gpus,
                '--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True)))
        for length in cfg['input_lens']:
            length_root=root/f'p{length}'
            length_root.mkdir()
            config={**cfg,'output_root':str(length_root/'sweep'),'input_len':length}
            if cfg.get('conservative_candidates_by_length'):
                config['conservative_candidates']=cfg['conservative_candidates_by_length'][str(length)]
            config_path=length_root/'config.json'
            write(config_path,config)
            command=[sys.executable,str(package/'sweep_decode_capacity.py'),
                '--config',str(config_path),'--repo',str(repo),'--model',model_id,
                '--gpus',args.gpus,'--lanes',','.join(lanes),'--attempt','capacity']
            status(f'p{length}/capacity','running')
            code=run(command,length_root/'sweep.log',cfg.get('length_timeout_s',172800))
            sweep_root=Path(config['output_root'])/model_id/(','.join(lanes).replace(',','_')+'-capacity')
            summary=json.loads((sweep_root/'queue_summary.json').read_text())
            if code:
                if summary['status']!='failed':raise RuntimeError('Sweep failed without lane failure classification')
                errors.append(dict(length=length,phase='capacity',failures=summary['failures']))
            status(f'p{length}/capacity',summary['status'])
            idle()
            guard_ready=length_root/'request_guard.json'
            with (length_root/'request_guard.log').open('x') as f:
                guard=subprocess.Popen([cfg['conda'],'run','--no-capture-output','-p',cfg['native_env'],
                    'python','-u',str(package/'decode_capacity_guard.py'),'--parent',str(os.getpid()),
                    '--ready',str(guard_ready)],cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(120):
                ensure_guard(guard_ready)
                if guard_ready.exists():break
                time.sleep(1)
            else:raise TimeoutError('Request reservation not ready')
            try:
                for lane in lanes:
                    ensure_guard(guard_ready)
                    capacity_path=sweep_root/lane/'capacity.json'
                    if not capacity_path.exists():continue
                    capacity=json.loads(capacity_path.read_text())
                    if capacity['status']!='completed':continue
                    conservative=capacity.get('selection')=='conservative_kv_slots'
                    batch=capacity['verified_concurrency'] if conservative else capacity['max_concurrency']
                    if not conservative and capacity['first_failed_concurrency']!=batch+1:
                        raise ValueError('Missing max+1 boundary')
                    engine,method,external=resolve_lane(cfg,lane)
                    dest=length_root/'requests'/lane
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    hp=json.loads((sweep_root/lane/f'bs{batch}'/'hyper_params.json').read_text())
                    hp_path=dest.with_suffix('.hyper_params.json')
                    write(hp_path,hp)
                    request_env=dict(env)
                    environment=external['env'] if external else cfg['vllm_env'] if engine=='vllm' else cfg['native_env']
                    launcher=[cfg['conda'],'run','--no-capture-output','-p',environment,'python','-u']
                    if external:
                        request_env.update(external.get('environment', {}))
                        request_env['PYTHONPATH']=os.pathsep.join([env['PYTHONPATH'], *external.get('pythonpath', [])])
                        if external.get('environment_kind')=='venv':
                            prefix=Path(environment)
                            launcher=[str(prefix/'bin/python'),'-u']
                            request_env['PATH']=str(prefix/'bin')+os.pathsep+request_env.get('PATH','')
                            request_env['VIRTUAL_ENV']=str(prefix)
                    command=launcher+['benchmark/efficiency/bench_probe.py','--engine',engine,
                        '--sparse-method',method,'--model-path',model['path'],
                        '--tensor-parallel-size',str(model['tp']),'--expert-parallel-size',str(model['ep']),
                        '--scenario','fixed','--prompt-lens',str(length),'--output-lens',str(cfg['output_len']),
                        '--batch-sizes',str(batch),'--seed','42','--prompt-length-jitter','0','--output-length-jitter','0',
                        '--max-num-batched-tokens',str(hp['max_num_batched_tokens']),
                        '--gpu-memory-utilization',str(cfg['gpu_memory_utilization']),
                        '--num-warmups',str(cfg['num_warmups']),'--num-iters',str(cfg['num_iters']),
                        '--hyper-params','@'+str(hp_path),'--output-dir',str(dest),'--monitor-gpus',args.gpus]
                    if 'sparse_prefill_score_mode' in hp:
                        command+=['--sparse-prefill-score-mode',hp['sparse_prefill_score_mode']]
                    if external:
                        kwargs=json.loads((sweep_root/lane/f'bs{batch}'/'engine_kwargs.json').read_text())
                        kwargs_path=dest.with_suffix('.engine_kwargs.json')
                        write(kwargs_path,kwargs)
                        command+=['--engine-kwargs','@'+str(kwargs_path),'--backend-label',external['backend_label']]
                    status(f'p{length}/{lane}/requests','running',batch=batch)
                    try:
                        code=run(command,dest.with_suffix('.log'),cfg['case_timeout_s'],
                                 guard_ready=guard_ready,run_env=request_env)
                    except subprocess.TimeoutExpired:
                        ensure_guard(guard_ready)
                        failure=dict(length=length,lane=lane,phase='request',
                                     error='request timeout',timeout_s=cfg['case_timeout_s'])
                        errors.append(failure)
                        status(f'p{length}/{lane}/requests','failed',**failure)
                        continue
                    if code:
                        errors.append(dict(length=length,lane=lane,phase='request',exitcode=code))
                        status(f'p{length}/{lane}/requests','failed',exitcode=code)
                        continue
                    state=json.loads((dest/'run_status.json').read_text())
                    samples=[json.loads(line) for line in (dest/'request_samples.jsonl').read_text().splitlines()]
                    if state['status']!='success' or len(samples)!=batch*cfg['num_iters'] or any(
                        s['status']!='success' or s['generated_tokens']!=cfg['output_len'] for s in samples):
                        raise RuntimeError(f'Incomplete request evidence: {dest}')
                    request=json.loads((dest/'summary.json').read_text())
                    decode=json.loads((sweep_root/lane/f'bs{batch}'/'performance.jsonl').read_text())
                    ensure_guard(guard_ready)
                    with (root/'results.jsonl').open('a') as f:
                        f.write(json.dumps(dict(input_tokens=length,lane=lane,
                            max_concurrency=None if conservative else batch, verified_concurrency=batch,
                            maximum_verified=not conservative,
                            capacity_artifact=str(capacity_path),request_artifact=str(dest),
                            request=request,decode=decode))+'\n')
                    status(f'p{length}/{lane}/requests','completed',batch=batch)
                ensure_guard(guard_ready)
            finally:
                stop_group(guard)
                guard=None
        write(root/'queue_status.json',dict(status='failed' if errors else 'completed',failures=errors))
        return int(bool(errors))
    except BaseException as error:
        write(root/'queue_status.json',dict(status='aborted',error=repr(error),failures=errors))
        raise
    finally:
        stop_group(active)
        stop_group(guard)


if __name__=='__main__':
    raise SystemExit(main())
