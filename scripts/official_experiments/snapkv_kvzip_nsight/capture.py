"""Annotate the existing probe; start CUDA capture only after its warmup."""
import contextlib
import functools
import json
import os
from pathlib import Path
import torch
from benchmark.efficiency import bench_probe as probe
from sparseengine.utils.profiler import profiler
from sparseengine.engine.sparse_methods import kvzip
from sparseengine.engine.cache_manager.methods.kvzip import KVzipCacheManager
from sparseengine.engine.model_runner import ModelRunner

@contextlib.contextmanager
def record(name):
    with torch.cuda.nvtx.range(name):
        yield
profiler.record = record

def annotate(owner, name, label):
    original = getattr(owner, name)
    @functools.wraps(original)
    def call(*args, **kwargs):
        with torch.cuda.nvtx.range(label):
            return original(*args, **kwargs)
    setattr(owner, name, call)

annotate(kvzip.KVzipRuntime, 'finish_step', 'kvzip.finish_step')
annotate(kvzip, 'select_kvzip_layers', 'kvzip.global_selection')
annotate(ModelRunner, '_run_auxiliary_prefill', 'kvzip.reconstruction_forward')
annotate(KVzipCacheManager, 'collect_prefill_attention_score', 'kvzip.score')
original_compact = KVzipCacheManager.compact_reconstruction

def compact(self, seq, keep):
    with torch.cuda.nvtx.range('kvzip.compact'):
        result = original_compact(self, seq, keep)
    with (Path(os.environ['RUN_ROOT'])/'kvzip_retention.jsonl').open('a') as f:
        f.write(json.dumps(dict(seq_id=seq.seq_id, context_length=seq.context_length,
                               per_layer_tokens=[x.numel() for x in keep]))+'\n')
    return result
KVzipCacheManager.compact_reconstruction = compact
original_monitor = probe._case_monitor
started = False

def monitor(*args, **kwargs):
    global started
    result = original_monitor(*args, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    started = True
    return result
probe._case_monitor = monitor
try:
    probe.main()
finally:
    if started:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
