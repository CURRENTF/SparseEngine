"""Create a Palu sidecar from a local, unquantized Hugging Face checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from sparseengine.models.layout import resolve_attention_qk_head_dim
from sparseengine.models.palu import factorize_grouped, tensor_digest


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def calibrate(model_path, path, samples, seqlen, ridge, device):
    """Collect uncentered activation covariance, as in Palu's whitening path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if samples <= 0 or seqlen <= 1 or not 0 < ridge < 1:
        raise ValueError('Calibration needs positive sample/token limits and ridge in (0,1).')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    token_batches = []
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError('Each calibration row must contain nonempty text.')
            ids = tokenizer(row['text'], truncation=True, max_length=seqlen, return_tensors='pt').input_ids
            if ids.numel() < 2:
                raise ValueError('Calibration sample has fewer than two tokens.')
            token_batches.append(ids)
            if len(token_batches) == samples:
                break
    if len(token_batches) != samples:
        raise ValueError(f'Calibration requires {samples} samples, found {len(token_batches)}.')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                torch_dtype='auto', attn_implementation='sdpa').to(device).eval()
    hidden = model.config.hidden_size
    grams = [torch.zeros(hidden, hidden, dtype=torch.float32, device=device) for _ in model.model.layers]
    counts = [0] * len(grams)
    handles = []
    def hook(index):
        def collect(module, inputs):
            x = inputs[0].reshape(-1, hidden).float()
            grams[index].addmm_(x.T, x)
            counts[index] += x.shape[0]
        return collect
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_pre_hook(hook(i)))
    try:
        for i, ids in enumerate(token_batches):
            model.model(input_ids=ids.to(device), use_cache=False)
            print(f'calibrated sample {i+1}/{samples}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    del model
    scales = []
    for gram, count in zip(grams, counts):
        if count <= 0:
            raise RuntimeError('Calibration did not execute an attention projection.')
        gram.div_(count)
        gram.diagonal().add_(ridge * gram.diagonal().mean())
        scales.append(torch.linalg.cholesky(gram).cpu())
    metadata = {'sha256': file_digest(path),
                'samples': samples, 'max_tokens_per_sample': seqlen, 'tokens_per_layer': counts,
                'ridge': ridge, 'input_ids_sha256': tensor_digest(torch.cat(token_batches, dim=1))}
    return scales, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--group-size', type=int, default=1)
    parser.add_argument('--rank-ratio', type=float, default=.5)
    parser.add_argument('--ranks-json', help='JSON file containing one [K rank, V rank] pair per layer; overrides ratio')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--calibration-jsonl', help='Whitening corpus: one {"text": ...} per line')
    parser.add_argument('--calibration-samples', type=int, default=32)
    parser.add_argument('--calibration-seqlen', type=int, default=1024)
    parser.add_argument('--calibration-ridge', type=float, default=1e-5)
    parser.add_argument('--uncalibrated', action='store_true', help='Explicit plain-SVD ablation; may severely damage quality')
    args = parser.parse_args()
    model, output = Path(args.model), Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if bool(args.calibration_jsonl) == args.uncalibrated:
        raise ValueError('Supply --calibration-jsonl or explicitly request --uncalibrated.')
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type not in ('llama', 'qwen3') or getattr(config, 'quantization_config', None):
        raise ValueError('Palu requires an unquantized Llama or Qwen3 checkpoint.')
    d, h, g = resolve_attention_qk_head_dim(config), config.num_key_value_heads, args.group_size
    if g <= 0 or h % g or not 0 < args.rank_ratio <= 1:
        raise ValueError('Invalid group size or rank ratio.')
    rank = max(16, int(g * d * args.rank_ratio) // 16 * 16)
    ranks = json.loads(Path(args.ranks_json).read_text()) if args.ranks_json else [[rank, rank] for _ in range(config.num_hidden_layers)]
    if len(ranks) != config.num_hidden_layers or any(len(pair) != 2 or any(type(r) is not int or r < 16 or r % 16 or r > min(g*d, 256) for r in pair) for pair in ranks):
        raise ValueError('Ranks must be multiples of 16, at most min(group_size*head_dim,256), with one K/V pair per layer.')
    files = sorted(model.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No safetensors weights in {model}')
    index = {}
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in index:
                    raise ValueError(f'Duplicate checkpoint tensor {key}')
                index[key] = path
    scales, calibration = (None, None)
    if args.calibration_jsonl:
        scales, calibration = calibrate(model, args.calibration_jsonl, args.calibration_samples,
                                       args.calibration_seqlen, args.calibration_ridge, args.device)
    tensors, fingerprints = {}, {}
    for layer, pair in enumerate(ranks):
        scale = None if scales is None else scales[layer].to(args.device)
        for kind, source, r in zip(('key', 'value'), ('k', 'v'), pair):
            name = f'model.layers.{layer}.self_attn.{source}_proj.weight'
            with safe_open(index[name], framework='pt', device='cpu') as handle:
                weight = handle.get_tensor(name)
            if weight.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f'Unsupported source dtype: {weight.dtype}')
            fingerprints[name] = tensor_digest(weight)
            a, b = factorize_grouped(weight.to(args.device), g, d, r, scale)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f'Nonfinite Palu factors for {name}.')
            tensors[f'layers.{layer}.{kind}_down'] = a.cpu()
            tensors[f'layers.{layer}.{kind}_up'] = b.cpu()
        print(f'factorized layer {layer + 1}/{len(ranks)} ranks={pair}', flush=True)
    manifest = {'format': 'sparsevllm.palu.v1', 'model': {
        'model_type': config.model_type, 'hidden_size': config.hidden_size,
        'num_attention_heads': config.num_attention_heads, 'num_key_value_heads': h,
        'num_hidden_layers': config.num_hidden_layers, 'head_dim': d},
        'group_size': g, 'ranks': ranks, 'source_weight_sha256': fingerprints,
        'factorization': 'grouped_svd_whitened' if scales is not None else 'grouped_svd_sqrt',
        'calibration': calibration, 'rank_allocation': 'explicit' if args.ranks_json else 'uniform',
        'torch_version': torch.__version__}
    output.mkdir(parents=True)
    save_file(tensors, str(output / 'palu.safetensors'))
    (output / 'palu.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
