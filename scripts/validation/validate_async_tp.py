"""Compare asynchronous TP with the synchronous execution reference.

This is a correctness runner, not a performance timing definition. All private
model and output paths are supplied by the caller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
import subprocess
import hashlib
import sys
from dataclasses import fields

import torch

from sparseengine import LLM, SamplingParams
from sparseengine.engine.async_scheduling.scheduler import AsyncScheduler


def run_case(llm, prompts, params, *, asynchronous, cancel_first=False, append_prompt=None,
             reset=True, sync_interlude=False):
    assert llm.is_finished()
    if hasattr(llm, '_async_scheduler'):
        del llm._async_scheduler
    if reset:
        llm.model_runner.call('reset_after_warmup')
    if asynchronous:
        llm._async_scheduler = AsyncScheduler(llm)
    torch.manual_seed(123)
    ids = [llm.add_request(p, sp) for p, sp in zip(prompts, params)]
    results = {}
    cached = {}
    steps = 0
    # Admit at the same scheduling boundary in both modes. Output retirement
    # is delayed in async mode and is not an equivalent arrival boundary:
    # different prefill batches can change low-precision scores and selection.
    schedule = llm.scheduler.schedule
    schedule_count = 0
    def schedule_with_arrival():
        nonlocal schedule_count
        if schedule_count == 2 and append_prompt is not None:
            ids.append(llm.add_request(append_prompt, params[-1]))
        schedule_count += 1
        return schedule()
    llm.scheduler.schedule = schedule_with_arrival
    try:
        while not llm.is_finished():
            # Exercise the drain -> synchronous recovery -> asynchronous transition
            # independently of allocator capacity, without changing model execution.
            if asynchronous and sync_interlude:
                if steps == 12:
                    llm._async_scheduler._submit = lambda: False
                elif steps == 12 + llm.config.async_max_inflight + 1:
                    del llm._async_scheduler._submit
            finished, _ = llm.step()
            cached.update(llm.last_step_prompt_cache_hits)
            for seq_id, tokens, logprobs, tops in finished:
                results[seq_id] = {'tokens': list(tokens), 'logprobs': list(logprobs), 'top_logprobs': tops, 'cached_tokens': cached.get(seq_id, 0)}
            steps += 1
            if steps == 1:
                if cancel_first:
                    llm.abort_request(ids[0])
            if steps > 1024:
                raise RuntimeError('Validation made no bounded progress')
    finally:
        llm.scheduler.schedule = schedule
    expected_ids = ids[1:] if cancel_first else ids
    assert set(results) == set(expected_ids)
    return [results[i] for i in expected_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--ep', type=int, default=2)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.7)
    parser.add_argument('--config', type=Path, help='JSON runtime overrides, including sparse method and budgets')
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'model': args.model, 'cases': [],
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'torch': str(torch.__version__), 'tp': args.tp, 'ep': args.ep, 'overrides': overrides,
              'command': sys.argv,
              'worktree_diff_sha256': hashlib.sha256(subprocess.check_output(
                  ['git', 'diff', 'HEAD', '--', 'src', 'scripts/validation/validate_async_tp.py'])).hexdigest()}

    destination.write_text(json.dumps(report, indent=2))
    llm = None
    try:
        options = dict(tensor_parallel_size=args.tp, expert_parallel_size=args.ep,
                       gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048,
                       max_num_seqs_in_batch=4, max_decoding_seqs=4, max_num_seqs_in_gpu=4,
                       max_num_batched_tokens=1024, engine_prefill_chunk_size=32,
                       enable_prefix_caching=True, prefix_cache_mode='radix',
                       async_scheduling=True, decode_graph_capture_sizes=[1, 2, 4])
        options.update(overrides)
        llm = LLM(args.model, **options)
        # HF configs acquire runtime objects after initialization; their repr
        # is no longer JSON serializable. Record resolved public scalar settings.
        report['resolved_config'] = {
            field.name: value for field in fields(llm.config)
            if isinstance(value := getattr(llm.config, field.name),
                          (str, int, float, bool, list, tuple, dict, type(None)))
        }
        graph_runner = llm.model_runner.decode_graph_runner
        startup_graph_plan = graph_runner.graph_plan()
        startup_captures = graph_runner.capture_count
        prompts = [[200+i % 31 for i in range(n)] for n in [19, 79, 113]]
        cases = [
            ('greedy', dict(temperature=0., ignore_eos=True), {}),
            ('penalties_logprobs', dict(temperature=0., ignore_eos=True, repetition_penalty=1.1,
                                       presence_penalty=.2, logprobs=3), {}),
            ('random_sampling', dict(temperature=.8, top_p=.9, top_k=20, ignore_eos=True), {}),
            ('cancellation', dict(temperature=0., ignore_eos=True), {'cancel_first': True}),
            ('new_arrival', dict(temperature=0., ignore_eos=True), {'append_prompt': prompts[0]}),
            ('sync_interlude', dict(temperature=0., ignore_eos=True, repetition_penalty=1.1,
                                    presence_penalty=.2), {'sync_interlude': True}),
        ]
        for name, options, extra in cases:
            params = [SamplingParams(max_tokens=n, **options) for n in [17, 11, 7]]
            reference = run_case(llm, prompts, params, asynchronous=False, **extra)
            actual = run_case(llm, prompts, params, asynchronous=True, **extra)
            assert [len(r['tokens']) for r in reference] == [len(r['tokens']) for r in actual]
            equal = [r['tokens'] for r in reference] == [r['tokens'] for r in actual]
            report['active_case'] = {'name': name, 'reference': reference, 'asynchronous': actual}
            destination.write_text(json.dumps(report, indent=2))
            if name != 'random_sampling':
                assert equal, f'{name}: greedy output mismatch'
            if name == 'penalties_logprobs':
                for ref, got in zip(reference, actual):
                    torch.testing.assert_close(torch.tensor(ref['logprobs']), torch.tensor(got['logprobs']), atol=.02, rtol=.02)
            report['cases'].append({'name': name, 'status': 'success', 'tokens_equal': equal,
                                    'reference': reference, 'asynchronous': actual})
            destination.write_text(json.dumps(report, indent=2))
        first = run_case(llm, [prompts[0]], [SamplingParams(max_tokens=1, temperature=0., ignore_eos=True)], asynchronous=False)[0]['tokens'][0]
        params = [SamplingParams(max_tokens=16, temperature=0., eos_token_ids=[first])]
        ref = run_case(llm, [prompts[0]], params, asynchronous=False)
        got = run_case(llm, [prompts[0]], params, asynchronous=True)
        assert ref[0]['tokens'] == got[0]['tokens'] == [first]
        report['cases'].append({'name': 'eos', 'status': 'success', 'reference': ref, 'asynchronous': got})
        if llm.config.enable_prefix_caching:
            extended = [[200+i % 31 for i in range(n)] for n in [384, 512]]
            params = [SamplingParams(max_tokens=8, temperature=0., ignore_eos=True)] * 2
            hot = []
            for asynchronous in [False, True]:
                run_case(llm, extended, params, asynchronous=asynchronous)
                hot.append(run_case(llm, extended, params, asynchronous=asynchronous, reset=False))
            assert all(row['cached_tokens'] >= 128 for rows in hot for row in rows)
            assert [r['tokens'] for r in hot[0]] == [r['tokens'] for r in hot[1]]
            report['cases'].append({'name': 'hot_radix_prefix', 'status': 'success',
                                    'reference': hot[0], 'asynchronous': hot[1]})
        assert graph_runner.graph_plan() == startup_graph_plan, 'Graph plan changed during execution'
        assert graph_runner.capture_count == startup_captures, 'Runtime graph capture occurred'
        assert graph_runner.recapture_count == 0, 'Runtime graph recapture occurred'
        report['rank0_graph_plan'] = startup_graph_plan
        report['operator_stats'] = llm.operator_runtime_stats()
        report['status'] = 'success'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2))
        if llm is not None:
            llm.exit()


if __name__ == '__main__':
    main()
