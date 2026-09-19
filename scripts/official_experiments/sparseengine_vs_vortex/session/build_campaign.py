"""Generate final experiment jobs once; do not launch GPUs or patch old runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)


def interpreter(backend, paths):
    if backend in ('native', 'vortex', 'vllm'):
        return [paths['conda'], 'run', '--no-capture-output', '-p', paths[backend + '_env'], 'python', '-u']
    if backend not in ('tangram', 'hisparse'):
        raise ValueError(f'Unknown backend: {backend}')
    return ['bash', '-ec', 'source "$1/bin/activate"; shift; exec python -u "$@"',
            'venv', paths[backend + '_env']]


def generate(config, paths, root, source, prepared, gpus, port_base=25500):
    """Return portable queue/config contents before performing any writes."""
    result, jobs = {}, []
    cases = json.loads((HERE.parent / 'configs/cases.json').read_text())['cases']
    def emit(relative, data):
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'Output escapes campaign root: {relative}')
        if relative in result:
            raise ValueError(f'Duplicate output: {relative}')
        result[relative] = data
        return str(root / relative)
    env = dict(PYTHONPATH=f'{source}:{source}/src', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_HOME=paths['cuda_home'], TOKENIZERS_PARALLELISM='false', TMPDIR=paths['scratch_root'])
    def queue(index, name, command, timeout, extra=None):
        gpu = gpus[index % len(gpus)]
        spec = dict(command=command, cwd=str(source), env={**env, 'SPARSEENGINE_MASTER_PORT': str(port_base + 100 + index), **(extra or {})},
                    timeout=timeout, continue_on_failure=True)
        path = emit(f'queues/{gpu}/jobs/{index:03d}_{name}.json', spec)
        jobs.append(dict(gpu=gpu, job=path, **spec))
    for index, arm in enumerate(config['quality']):
        name, model, backend = arm['id'], config['models'][arm['model']], arm['backend']
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError(f'Invalid arm identity: {name}')
        hp = copy.deepcopy(config['profiles'][arm['profile']])
        model_path = str(Path(paths['model_root']) / model['name'])
        engine = 'sparseengine' if backend == 'native' else ('vllm' if backend in ('vllm', 'tangram') else 'sglang-http')
        command = interpreter('native' if engine == 'sglang-http' else backend, paths) + [
            'benchmark/long_bench_v2/pred.py', '--model-path', model_path, '--sparse-method', arm['method'],
            '--engine', engine, '--data-path', paths['dataset'], '--official-length', 'medium',
            '--max-model-len', str(model['max_len']), '--max-new-tokens', str(config['max_new_tokens']),
            '--batch-size', '1', '--seed', str(config['seed'])]
        port = port_base + index
        if engine == 'sparseengine':
            command += ['--hyper-param-json', emit(f'configs/{name}.json', {**config['native_runtime'], **hp})]
        elif engine == 'vllm':
            command += ['--engine-kwargs', emit(f'configs/{name}.json', hp)]
        else:
            command += ['--server-url', f'http://127.0.0.1:{port}']
        dest = root / 'quality' / name
        qspec = emit(f'configs/{name}.job.json', dict(command=command, cwd=str(source), output=str(dest),
                     prepared=str(prepared / model['name'] / 'prepared_samples.json'), timeout=36000))
        runner = ['python3', str(source / HERE.relative_to(REPO) / 'run_quality.py'), qspec]
        if backend in ('hisparse', 'vortex'):
            server_env, server_cwd = {}, str(REPO)
            if backend == 'hisparse':
                server = interpreter(backend, paths) + ['-m', 'sglang.launch_server', '--model-path', model_path,
                    '--host', '127.0.0.1', '--port', str(port), '--tp', '1', '--dtype', 'bfloat16', '--mem-fraction-static', '0.9',
                    '--context-length', str(model['max_len']), '--max-running-requests', '1', '--chunked-prefill-size', '8192',
                    '--max-prefill-tokens', '8192', '--disable-radix-cache', '--attention-backend', 'fa3', '--page-size', '16',
                    '--enable-hisparse', '--hisparse-config', json.dumps(hp), '--random-seed', str(config['seed']),
                    '--cuda-graph-config', json.dumps({'prefill': {'backend': 'disabled'}, 'decode': {'backend': 'breakable', 'bs': [1]}})]
            else:
                key = 'qwen' if arm['model'] == 'qwen4' else arm['model']
                case, = [c for c in cases if c['model_key'] == key and c['method'].lower() == arm['method'] and c['engine'].startswith('Vortex')]
                vortex = copy.deepcopy(case['vortex_config'])
                vortex.update(max_seq_lens=model['max_len'], compilation_cache_dir=str(root / 'cache' / name))
                vortex = {k: v.replace('${VORTEX_REPO}', paths['vortex_repo']) if isinstance(v, str) else v for k, v in vortex.items()}
                argv = list(case['server_config']['server_args'])
                for flag, value in {'--model-path': model_path, '--vortex-config': json.dumps(vortex), '--context-length': str(model['max_len']),
                                    '--max-running-requests': '1', '--cuda-graph-max-bs': '1', '--port': str(port), '--random-seed': str(config['seed'])}.items():
                    argv[argv.index(flag) + 1] = value
                server_config = emit(f'configs/{name}.server.json', {'server_args': argv})
                server = interpreter(backend, paths) + [str(source / 'scripts/official_experiments/sparseengine_vs_vortex/vortex_server.py'), server_config]
                overlay, fork = paths['vortex_overlay'], paths['vortex_repo']
                server_cwd = fork
                server_env = dict(PYTHONPATH=f'{overlay}:{overlay}/nvidia_cutlass_dsl/dsl_packages:{fork}:{fork}/third_party/sglang/v0.5.9/sglang/python',
                    FLASHINFER_CUBIN_DIR=overlay + '/cubins', VORTEX_CUDA_MLA_SM90_BUILD_DIR=str(root / 'cache/vortex_mla'),
                    SGLANG_ENABLE_TORCH_COMPILE='0', MAX_JOBS='8')
            spec = emit(f'configs/{name}.server_job.json', dict(output=str(dest), server_command=server, server_cwd=server_cwd,
                         server_env=server_env, server_url=f'http://127.0.0.1:{port}', quality_spec=qspec))
            runner = ['python3', str(source / HERE.relative_to(REPO) / 'serve_quality.py'), spec]
        queue(index, name, runner, 41000)
    index = len(config['quality'])
    for lane in config['efficiency32']:
        for batch in lane['batches']:
            name = f'32k_{lane["label"]}_b{batch}'
            hyper = emit(f'configs/{name}.hyper.json', lane['hyper_params'])
            backend = lane['backend']
            engine = {'native': 'sparseengine', 'tangram': 'vllm', 'hisparse': 'hisparse'}[backend]
            command = interpreter(backend, paths) + ['benchmark/microbench.py', '--model_path',
                str(Path(paths['model_root']) / config['models']['qwen4']['name']), '--engine', engine, '--methods', lane['method'],
                '--lengths', '32768', '--output_len', '512', '--batch_sizes', str(batch), '--max_model_len_override', '33408',
                '--require_full_decode_batch', '--synchronize_step_timing', '--decode_warmup_steps_after_full', '8',
                '--hyper_params', '@' + hyper, '--output_dir', str(root / 'efficiency32' / name)]
            if backend != 'native':
                command += ['--backend_label', lane['label'], '--engine_kwargs', '@' + emit(f'configs/{name}.engine.json', lane['engine_kwargs'])]
            queue(index, name, command, 7200, {'VLLM_ENABLE_V1_MULTIPROCESSING': '0'} if backend == 'tangram' else None)
            index += 1
    for gpu in gpus:
        emit(f'queues/{gpu}/STOP', {'reason': 'All final jobs generated; release guard after drain'})
    emit('queued_commands.json', jobs)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE / 'campaign.json')
    p.add_argument('--paths', type=Path, required=True, help='JSON paths: model_root, dataset, conda, native/vllm/vortex/tangram/hisparse_env, vortex_repo/overlay, scratch_root, cuda_home')
    p.add_argument('--root', type=Path, required=True, help='Fresh external run directory')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--source', type=Path, default=REPO, help='Checkout to execute directly (default: this repository)')
    p.add_argument('--gpus', required=True, help='Comma-separated physical GPU indices; generation does not reserve them')
    p.add_argument('--port-base', type=int, default=25500)
    a = p.parse_args()
    config, paths = json.loads(a.config.read_text()), json.loads(a.paths.read_text())
    gpus = a.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not x.isdigit() for x in gpus):
        raise ValueError('Expected distinct physical GPU indices')
    if not 1024 <= a.port_base <= 65000:
        raise ValueError('Port base must leave room for per-job ports')
    for key in ('model_root', 'dataset', 'conda', 'native_env', 'vllm_env', 'vortex_env', 'tangram_env', 'hisparse_env', 'vortex_repo', 'vortex_overlay', 'cuda_home'):
        if not Path(paths[key]).is_absolute() or not Path(paths[key]).exists():
            raise ValueError(f'Missing absolute input path: {key}')
    if not Path(paths['scratch_root']).is_absolute():
        raise ValueError('scratch_root must be an absolute external path')
    for model in config['models'].values():
        if not (Path(paths['model_root']) / model['name']).is_dir():
            raise FileNotFoundError(model['name'])
        if not (a.prepared / model['name'] / 'prepared_samples.json').is_file():
            raise FileNotFoundError(f'Missing prepared cohort for {model["name"]}')
    source = a.source.resolve(strict=True)
    plan = generate(config, paths, a.root.resolve(), source, a.prepared.resolve(), gpus, a.port_base)
    a.root.mkdir(parents=True, exist_ok=False)
    for relative, data in plan.items():
        write(a.root / relative, data)
    write(a.root / 'campaign.json', config)
    write(a.root / 'plan.json', dict(source=str(source),
          git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
          git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()),
          prepared=str(a.prepared.resolve()), config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
          paths=paths, gpus=gpus, status='prepared_not_launched'))
    print(f'Prepared {len(plan["queued_commands.json"])} jobs; launch via ../run_queue.py on idle GPUs.')


if __name__ == '__main__':
    main()
