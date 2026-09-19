"""Artifact-only progress/quality aggregation and canonical raw-stage validation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO),str(REPO/'scripts/official_experiments/sparse_decode_efficiency')]
from benchmark.long_bench_v2.contracts import aggregate_results
from plot_decode_capacity import validate_measurement

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def summarize(ROOT, output, config, prepared, validated128):
    quality=[]
    queues=list((ROOT/'queues').iterdir())
    terminal=bool(queues) and all((q/'queue_exit.json').exists() for q in queues)
    for arm in sorted((ROOT/'quality').glob('*')):
        full=arm/'full'
        if not full.exists():
            smoke=arm/'smoke/run_status.json'
            state=read(smoke)['status'] if smoke.exists() else 'not_started'
            quality.append({'case':arm.name,'status':'smoke_'+state if not terminal else 'incomplete',
                'accuracy':None,'evaluated_samples':0,'expected_samples':198,'parse_failed':None,'artifact':str(arm)})
            continue
        status=read(full/'run_status.json') if (full/'run_status.json').exists() else {'status':'initializing'}
        record={'case':arm.name,'status':status['status'],'accuracy':None,'evaluated_samples':0,'expected_samples':198,'parse_failed':None,'artifact':str(full)}
        if (full/'progress.json').exists():record['evaluated_samples']=read(full/'progress.json')['finished']
        if status['status']=='completed':
            metrics=read(full/'aggregate_metrics.json');sample_rows=rows(full/'sample_results.jsonl')
            resolved=read(full/'resolved_config.json')
            expected=rows(prepared/Path(resolved['model_path']).name/'dataset.jsonl')
            if len(sample_rows)!=198 or {r['_id'] for r in sample_rows}!={r['_id'] for r in expected}:
                raise ValueError(f'Quality sample coverage mismatch: {arm}')
            recomputed=aggregate_results(sample_rows)
            if recomputed['status']!='success' or any(metrics[k]!=recomputed[k] for k in recomputed):
                raise ValueError(f'Quality raw metrics mismatch: {arm}')
            record.update(accuracy=metrics['accuracy'],evaluated_samples=metrics['evaluated_samples'],parse_failed=metrics['parse_failed_samples'])
        quality.append(record)
    stage=[];curves={}
    for path in sorted((ROOT/'efficiency32').glob('*/performance.jsonl')):
        data=rows(path)
        if len(data)!=1 or data[0].get('status')!='success':continue
        raw=data[0];batch=raw['batch_size']
        point=validate_measurement(path,batch,{'input_len':32768,'output_len':512})
        lane=raw.get('backend_label') if raw['engine']!='sparseengine' else 'sengine-'+raw['method']
        if lane=='hisparse-quest-pr-series':lane='hisparse-quest'
        stage.append({'lane':lane,**point})
        key=(lane,batch)
        if key in curves:raise ValueError(f'Duplicate successful stage point: {key}')
        curves[key]=point
    write(output/'quality.json',quality);write(output/'efficiency32.json',stage)
    def quality_column(name):
        found=next((x for x in quality if x['case']==name),None)
        return {'quality_accuracy':found['accuracy'] if found else None,
                'quality_status':found['status'] if found else 'queued',
                'quality_samples':found['evaluated_samples'] if found else 0,
                'quality_artifact':found['artifact'] if found else str(ROOT/'quality'/name/'full')}
    mapping32={lane['lane']:lane['quality'] for lane in config['efficiency32']}
    comparison32=[{**x,**quality_column(mapping32[x['lane']]),
                   'comparison_note':'HiSparse quality uses fixed ratio, not fixed token budget' if x['lane']=='hisparse-quest' else ''} for x in stage]
    comparison128=[]
    for item in validated128:
        name=config['quality128'][item['model']][item['lane']]
        comparison128.append({**item,**quality_column(name),
            'comparison_note':'quality is TP1; GLM efficiency is original TP2/EP2; HiSparse uses fixed ratio'})
    for label,values in [('comparison32',comparison32),('comparison128',comparison128)]:
        write(output/(label+'.json'),values)
        if values:
            with (output/(label+'.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    if quality:
        with (output/'quality.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(quality[0]));writer.writeheader();writer.writerows(quality)
    # Fixed-ratio upstream QuEST is not a fixed-token budget on variable contexts.
    from ctypes import c_float
    budgets=[]
    for model,ratio,recent in [('Qwen3-4B-Instruct-2507',252/2044,4),('Qwen3-30B-A3B-Instruct-2507-FP8',96/8160,32)]:
        for item in rows(prepared/model/'dataset.jsonl'):
            pages=(item['prompt_tokens']+15)//16;history=pages-recent
            keep=int(c_float(c_float(history).value*c_float(ratio).value).value)
            budgets.append({'model':model,'_id':item['_id'],'prompt_tokens':item['prompt_tokens'],
                            'selected_token_capacity_at_prompt_end':16*(max(keep,1)+recent),
                            'budget_kind':'fixed upstream ratio; nominal page capacity, not traced realized per-layer count'})
    write(output/'hisparse_quality_budget_audit.json',budgets)
    expected_names=[arm['id'] for arm in config['quality']]
    write(output/'quality_attempts.json',quality)
    indexed={x['case']:x for x in quality}
    public_quality=[indexed.get(name,{'case':name,'status':'not_completed' if terminal else 'queued',
                    'accuracy':None,'evaluated_samples':0,'expected_samples':198,'parse_failed':None,
                    'artifact':str(ROOT/'quality'/name/'full')}) for name in expected_names]
    write(output/'quality.json',public_quality)
    with (output/'quality.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(public_quality[0]));writer.writeheader();writer.writerows(public_quality)
    done={x['case'] for x in quality if x['status']=='completed'}
    missing=sorted(set(expected_names)-done)
    expected_points=sum(len(lane['batches']) for lane in config['efficiency32'])
    all_complete=not missing and len(stage)==expected_points
    write(output/'campaign_status.json',dict(status=('completed' if all_complete else 'incomplete') if terminal else 'running',updated=time.time(),
         completed_quality=sum(x['status']=='completed' for x in quality),active_quality=sum(x['status']=='running' for x in quality),
         expected_quality=len(expected_names),pending_or_failed_quality=missing,
         validated_efficiency32_points=len(stage),reused_efficiency128_points=len(validated128)))
    print(json.dumps(read(output/'campaign_status.json')),flush=True)
    return terminal

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh summary directory; never overwrite a published summary')
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('campaign.json'))
    p.add_argument('--prepared',type=Path,help='Defaults to ROOT/prepared_v2 for historical campaigns')
    p.add_argument('--efficiency128',type=Path,help='Defaults to ROOT/efficiency128/plot_data.json')
    p.add_argument('--watch',action='store_true',help='Bounded artifact polling only; no automatic repairs')
    a=p.parse_args()
    config=read(a.config)
    prepared=a.prepared or a.root/'prepared_v2'
    data=read(a.efficiency128 or a.root/'efficiency128/plot_data.json')
    validated=[]
    for curve in data['curves']:
        cfg=dict(data['config'])
        override=cfg.get('curve_protocols',{}).get(curve['model'],{}).get(curve['lane'],{})
        cfg.update({k:v for k,v in override.items() if k in ('input_len','output_len')})
        for point in curve['points']:
            actual=validate_measurement(Path(point['artifact']),point['concurrency'],cfg)
            if actual!=point:raise ValueError(f'Reused point changed: {curve["model"]}/{curve["lane"]}')
            validated.append({'model':curve['model'],'lane':curve['lane'],**actual})
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'raw_revalidation128.json',dict(status='success',points=validated))
    for _ in range(720 if a.watch else 1):
        if summarize(a.root,a.output,config,prepared,validated) or not a.watch:break
        time.sleep(60)
    else:raise TimeoutError('Summary watcher exceeded 12 hours')


if __name__=='__main__':
    main()
