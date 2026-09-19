"""Resolve the recorded paper cases into portable guarded-queue inputs; no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import string
import subprocess

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO = EXPERIMENT_DIR.parents[2]


def resolve(value, variables):
    if isinstance(value, dict):
        return {key: resolve(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, variables) for item in value]
    return string.Template(value).substitute(variables) if isinstance(value, str) else value


def build_outputs(manifest, variables, selected_models, gpus, ports):
    """Build all files before writing, so a bad template cannot publish half a plan."""
    outputs = {}
    seen_ids = set()
    for case in manifest['cases']:
        model = case['model_key']
        if model not in selected_models:
            continue
        case_id = case['id']
        if not isinstance(case_id, str) or not case_id or any(
                not (char.isascii() and (char.isalnum() or char in '_-')) for char in case_id):
            raise ValueError(f'Invalid case id: {case_id}')
        if case_id in seen_ids:
            raise ValueError(f'Duplicate case id: {case_id}')
        seen_ids.add(case_id)
        relative = Path(case['directory'])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2 or relative.parts[0] != model:
            raise ValueError(f'Invalid case directory: {relative}')
        identifier = relative.name
        hyper = f'configs/{case["id"]}.hyper.json'
        server = f'configs/{case["id"]}.server.json'
        values = {**variables, 'MODEL_KEY': model, 'GPU': str(gpus[model]),
                  'NATIVE_PORT': str(ports[model][0]), 'VORTEX_PORT': str(ports[model][1]),
                  'HYPER_PARAMS_FILE': str(Path(variables['OUTPUT_ROOT']) / hyper),
                  'SERVER_CONFIG_FILE': str(Path(variables['OUTPUT_ROOT']) / server)}
        if 'vortex_config' in case:
            values['VORTEX_CONFIG_JSON'] = json.dumps(resolve(case['vortex_config'], values))
            outputs[server] = resolve(case['server_config'], values)
        else:
            outputs[hyper] = resolve(case['hyper_params'], values)
        job_path = f'{model}/jobs/{identifier}.json'
        if job_path in outputs:
            raise ValueError(f'Duplicate case: {case["id"]}')
        outputs[job_path] = resolve(case['job'], values)
    if not outputs:
        raise ValueError('No cases selected')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=EXPERIMENT_DIR / 'configs/cases.json')
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--vortex-repo', type=Path, required=True)
    parser.add_argument('--vortex-overlay', type=Path, required=True,
                        help='Prepared FlashInfer 0.6.17 overlay with CUTLASS DSL')
    parser.add_argument('--native-env', type=Path, required=True)
    parser.add_argument('--vortex-env', type=Path, required=True)
    parser.add_argument('--conda', default=shutil.which('conda'))
    parser.add_argument('--cuda-home', type=Path, default=os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['qwen', 'glm'], default=['qwen', 'glm'])
    parser.add_argument('--qwen-gpu', type=int)
    parser.add_argument('--glm-gpu', type=int)
    parser.add_argument('--qwen-native-port', type=int, default=24435)
    parser.add_argument('--glm-native-port', type=int, default=24436)
    parser.add_argument('--qwen-vortex-port', type=int, default=30405)
    parser.add_argument('--glm-vortex-port', type=int, default=30406)
    args = parser.parse_args()
    selected = set(args.models)
    gpus = {'qwen': args.qwen_gpu, 'glm': args.glm_gpu}
    if any(gpus[key] is None or gpus[key] < 0 for key in selected) or len({gpus[key] for key in selected}) != len(selected):
        raise ValueError('Specify a distinct nonnegative physical GPU ID for each selected model')
    ports = {'qwen': (args.qwen_native_port, args.qwen_vortex_port),
             'glm': (args.glm_native_port, args.glm_vortex_port)}
    all_ports = [port for model in selected for port in ports[model]]
    if len(set(all_ports)) != len(all_ports) or any(not 1 <= port <= 65535 for port in all_ports):
        raise ValueError('Server/master ports must be valid and distinct')
    conda = shutil.which(args.conda) if args.conda else None
    if conda is None:
        raise FileNotFoundError('Conda executable not found; pass --conda')
    for path in (args.model_root, args.vortex_repo, args.native_env, args.vortex_env,
                 args.cuda_home, args.vortex_overlay / 'flashinfer',
                 args.vortex_overlay / 'nvidia_cutlass_dsl/dsl_packages'):
        if not path.is_dir():
            raise FileNotFoundError(path)
    model_names = {'qwen': 'Qwen3-4B-Instruct-2507', 'glm': 'GLM-4.7-Flash'}
    for key in selected:
        path = args.model_root / model_names[key] / 'config.json'
        if not path.is_file():
            raise FileNotFoundError(path)
    variables = {'SPARSEENGINE_REPO': str(REPO), 'EXPERIMENT_DIR': str(EXPERIMENT_DIR),
                 'MODEL_ROOT': str(args.model_root.resolve()), 'VORTEX_REPO': str(args.vortex_repo.resolve()),
                 'VORTEX_OVERLAY': str(args.vortex_overlay.resolve()), 'CONDA': conda,
                 'NATIVE_ENV': str(args.native_env.resolve()), 'VORTEX_ENV': str(args.vortex_env.resolve()),
                 'OUTPUT_ROOT': str(args.output_root.resolve())}
    manifest = json.loads(args.cases.read_text())
    outputs = build_outputs(manifest, variables, selected, gpus, ports)
    for name, value in outputs.items():
        if '/jobs/' in name:
            value['env']['CUDA_HOME'] = str(args.cuda_home.resolve())
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        path = args.output_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')
    commands = []
    for model in sorted(selected):
        (args.output_root / model / 'STOP').write_text('Release guard after all queued jobs complete.\n')
        commands.append(['python3', str(EXPERIMENT_DIR / 'run_queue.py'), '--root',
                         str(args.output_root.resolve() / model), '--gpu', str(gpus[model]),
                         '--native-env', str(args.native_env.resolve()), '--conda', conda])
    metadata = {'status': 'prepared_not_run', 'commands': commands,
                'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'note': 'Preparation does not reserve GPUs. Run each queue in a dedicated tmux session; its guard checks idleness and ownership.'}
    (args.output_root / 'execution_plan.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
