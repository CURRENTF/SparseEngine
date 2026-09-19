"""Prepare a common, untruncated native-context medium cohort and exclusion audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[4]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path(__file__).with_name('campaign.json'))
p.add_argument('--model-root', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--output', type=Path, required=True, help='Fresh prepared-cohort directory')
a = p.parse_args()
config = json.loads(a.config.read_text())
ROOT = a.output
ROOT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(REPO), str(REPO/'src')]
from transformers import AutoTokenizer
from benchmark.long_bench.pred import build_chat
from benchmark.long_bench_v2.contracts import load_dataset, render_prompt, file_sha256
from benchmark.long_bench_v2.pred import _identity, _write_json, _write_jsonl

DATA = a.data
PROMPT = REPO/'benchmark/long_bench_v2/upstream/prompts/0shot.txt'
rows = [r for r in load_dataset(DATA) if r['length']=='medium']
assert len(rows)==215
template = PROMPT.read_text()
models = [(m['name'], m['max_len']) for m in config['models'].values()]

def prepare(pair):
    model, capacity = pair
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_root / model),trust_remote_code=True)
    result = []
    for i,row in enumerate(rows):
        prompt = build_chat(tokenizer,render_prompt(template,row),'longbench_v2',no_chat_template=False,thinking_mode='off')
        ids = tokenizer.encode(prompt,add_special_tokens=bool(tokenizer.bos_token is not None and not prompt.startswith(tokenizer.bos_token)))
        result.append(dict(index=i,source_index=i,sample=row,prompt=prompt,prompt_token_ids=ids,prompt_tokens=len(ids),token_bucket='medium'))
        if (i+1)%25==0: print(model,i+1,flush=True)
    return model,capacity,result

with ThreadPoolExecutor(max_workers=3) as pool:
    prepared = list(pool.map(prepare,models))
eligible = set.intersection(*[{x['sample']['_id'] for x in items if x['prompt_tokens']<=capacity-config['max_new_tokens']}
                             for _,capacity,items in prepared])
if not eligible: raise RuntimeError('No common in-range medium samples')
audit = []
for i,row in enumerate(rows):
    lengths = {m:items[i]['prompt_tokens'] for m,_,items in prepared}
    audit.append({'_id':row['_id'],'status':'selected' if row['_id'] in eligible else 'skipped_by_policy',
                  'prompt_tokens':lengths,'reason':None if row['_id'] in eligible else 'exceeds at least one native model context; no truncation or RoPE changes'})
_write_jsonl(ROOT/'cohort_audit.jsonl',audit)
for model,capacity,items in prepared:
    selected = sorted([x for x in items if x['sample']['_id'] in eligible],key=lambda x:hashlib.sha256(f"{config['seed']}:{x['sample']['_id']}".encode()).hexdigest())
    for i,x in enumerate(selected): x['index']=i
    identity = dict(data_sha256=file_sha256(DATA),prompt_template_sha256=file_sha256(PROMPT),
                    tokenizer_path=str(a.model_root / model),seed=config['seed'],no_chat_template=False,
                    official_length='medium',token_buckets=[dict(name='medium',min_prompt_tokens=1,max_prompt_tokens=capacity-config['max_new_tokens'],samples=len(selected))])
    out = ROOT/model
    _write_json(out/'prepared_samples.json',{'identity':identity,'samples':selected})
    _write_jsonl(out/'dataset.jsonl',[_identity(x) for x in selected])
    _write_json(out/'run_status.json',dict(status='prepared',samples=len(selected),official_medium_samples=215,
                                        excluded=215-len(selected),min_tokens=min(x['prompt_tokens'] for x in selected),max_tokens=max(x['prompt_tokens'] for x in selected)))
    print(model,'prepared',len(selected),flush=True)
