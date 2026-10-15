#!/usr/bin/env python3
"""Real BF16 chain validation. Pass model/output paths; select an idle GPU externally."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from sparseengine import LLM, SamplingParams
from sparseengine.engine.chain_cache import ChainState


def check_state(manager, coordinator):
    chain_reserved = manager._outstanding()
    decode_reserved = coordinator.decode_reservations.outstanding()
    reserved = tuple(chain + decode_reserved.get(f"layer_{layer}", 0)
                     for layer, chain in zip(manager.kv_transformer_layer_indices(), chain_reserved))
    manager.allocator.assert_invariants(reserved)
    for sid, row in manager.allocator.seq_rows.items():
        np.testing.assert_array_equal(manager.valid_lengths[:, row].cpu(), manager.allocator.lengths[:, row])
        for l, h in np.ndindex(manager.allocator.lengths[:, row].shape):
            n = manager.allocator.lengths[l, row, h]
            np.testing.assert_array_equal(manager.slot_table[l, row, h, :n].cpu(),
                                          manager.allocator.slots[l, row, h, :n])
            np.testing.assert_array_equal(manager.original_positions[l, row, h, :n].cpu(),
                                          manager.allocator.positions[l, row, h, :n])
    return dict(free=manager.allocator.free_counts.tolist(), reserved=list(reserved),
                allocated=manager.allocator.lengths.sum(axis=(1, 2)).tolist(),
                head_lengths={str(sid): manager.allocator.lengths[:, row].tolist()
                              for sid, row in manager.allocator.seq_rows.items()},
                chains={r.chain_id: dict(state=r.state.value, seq_id=r.seq_id,
                                         physical=list(r.physical_slots_by_layer))
                        for r in coordinator.index.records.values()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--reference", action="store_true", help="Use the independent attention oracle")
    parser.add_argument("--materialized-reference", action="store_true",
                        help="Independent CPU-ledger addressing, ordinary BF16 attention")
    parser.add_argument("--check-materialized-attention", action="store_true",
                        help="Check actual and materialized attention against FP32 on identical inputs")
    parser.add_argument("--selection-reference", help="Use the same original-position selections as this run")
    parser.add_argument("--token-layout", action="store_true", help="Old SnapKV with compression disabled")
    parser.add_argument("--prompt-length", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--profile", action="store_true",
                        help="Synchronized component diagnostics; disable oracle/logit copies")
    parser.add_argument("--budget-policy", choices=("fixed", "adakv"), default="fixed")
    parser.add_argument("--head-budget", type=int,
                        help="Average total AdaKV budget, or uniform fixed head budgets")
    parser.add_argument("--adakv-alpha", type=float, default=0.2)
    args = parser.parse_args()
    torch.manual_seed(42)
    if sum((args.reference, args.materialized_reference, args.check_materialized_attention, args.profile)) > 1:
        parser.error("Choose one reference/check mode or profiling.")
    if args.budget_policy == "adakv" and (args.no_compression or args.token_layout):
        parser.error("AdaKV probe requires head-slot compression.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(args.model)
    budgets = [[16 + 8 * ((l + h) % 4) for h in range(hf.num_key_value_heads)]
               for l in range(hf.num_hidden_layers)]
    average_budget = 28 if args.head_budget is None else args.head_budget
    if args.head_budget is not None or args.budget_policy == "adakv":
        if average_budget < 6:
            parser.error("This probe protects 2 sink + 4 recent tokens; head-budget must be >= 6.")
        budgets = [[average_budget] * hf.num_key_value_heads for _ in range(hf.num_hidden_layers)]
    config = dict(sparse_method="snapkv", kv_cache_layout="head_slots",
                  head_slot_budgets=None if args.no_compression else budgets,
                  sink_keep_tokens=2, recent_keep_tokens=4, observation_window_size=16,
                  sparse_prefill_score_mode="probability", pool_kernel_size=1,
                  max_model_len=max(512, args.prompt_length + 3 * args.max_tokens + 256),
                  max_num_batched_tokens=128, engine_prefill_chunk_size=64,
                  max_num_seqs_in_batch=2, max_decoding_seqs=2, max_num_seqs_in_gpu=2,
                  gpu_memory_utilization=0.3, enable_prefix_caching=True,
                  decode_graph=not args.eager, decode_graph_capture_sampling=False,
                  decode_graph_capture_sizes=[1, 2], throughput_log_interval_s=0.0)
    if args.budget_policy == "adakv":
        config.update(sparse_method="adakv", head_slot_budgets=None,
                      adakv_alpha=args.adakv_alpha, decode_keep_tokens=average_budget - 6)
    if (args.reference or args.materialized_reference or args.check_materialized_attention) and not args.eager:
        raise ValueError("The independent Python oracle requires --eager.")
    if args.token_layout:
        if not args.no_compression:
            raise ValueError("Old token-layout comparison requires --no-compression.")
        config.update(kv_cache_layout="token", head_slot_budgets=None, decode_keep_tokens=8192)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2))
    os.environ["SPARSEENGINE_DEBUG_RUNTIME"] = "1"
    llm = LLM(args.model, **config)
    manager = llm.model_runner.cache_manager
    coordinator = llm.model_runner.runtime_state.chain_cache_coordinator
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.max_tokens)
    records, logits, trace, selections, score_checks, requests = [], [], [], {}, [], []
    selection_scores, allocation_checks = {}, []
    selection_counts = {}
    seq_labels = {}
    current_round = 0
    from sparseengine.utils.profiler import profiler
    profile_samples = []
    saved_record = profiler.record
    if args.profile:
        profiler.reset()
        profiler.device_sync = True
        profiler.set_enabled(True)
        measured = {"head_slot_metadata", "head_slot_scoring", "head_slot_compaction",
                    "head_slot_append", "head_slot_selection", "model_prefill", "model_decode"}
        @contextmanager
        def record_component(name):
            if name not in measured:
                yield
                return
            before = profiler.times[name]
            with saved_record(name):
                yield
            profile_samples.append(dict(component=name, round=current_round,
                                        elapsed_s=profiler.times[name] - before,
                                        status="success"))
        profiler.record = record_component
        original_append = manager._append
        def timed_append(*args, **kwargs):
            with profiler.record("head_slot_append"):
                return original_append(*args, **kwargs)
        manager._append = timed_append
    graph = llm.model_runner.decode_graph_runner
    counter_names = ("capture_count", "replay_count", "eager_static_count", "force_eager_count", "recapture_count")
    counters_before = {name: getattr(graph, name, 0) for name in counter_names}
    graph_ptrs = {str(k): v.decode_state.inputs.data_ptrs() for k, v in graph._graphs.items()} if graph else {}
    from head_slot_reference import attention as reference_attention, scores as reference_scores
    materialized = None
    attention_checks = []
    if args.materialized_reference or args.check_materialized_attention:
        from head_slot_reference import MaterializedReference
        materialized = MaterializedReference(manager)
    owner = llm.model_runner.model.model.layers[0].self_attn.attn.full_attention_provider
    if args.check_materialized_attention:
        def checked_attention(original, is_prefill):
            def checked(q, view, **kwargs):
                actual = original(q, view, **kwargs)
                cu = kwargs["qo_indptr"] if is_prefill else torch.arange(q.shape[0] + 1, device=q.device)
                expected = reference_attention(q, view, cu, manager._positions)
                dense = materialized(q, view, cu, manager._positions, is_prefill=is_prefill)
                # Same BF16-vs-FP32 numerical contract as the independent CUDA
                # attention tests; no cumulative full-network error is allowed
                # to obscure which operation produced a discrepancy.
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                torch.testing.assert_close(dense, expected, atol=0.02, rtol=0.02)
                attention_checks.append(dict(status="success", prefill=is_prefill,
                    actual_fp32_max_abs_error=float((actual.float() - expected.float()).abs().max()),
                    materialized_fp32_max_abs_error=float((dense.float() - expected.float()).abs().max()),
                    actual_materialized_max_abs_error=float((actual.float() - dense.float()).abs().max())))
                return actual
            return checked
        owner.prefill_op.run = checked_attention(owner.prefill_op.run, True)
        owner.decode_op.run = checked_attention(owner.decode_op.run, False)
    if args.reference or args.materialized_reference:
        def prefill_reference(q, view, **kwargs):
            if materialized:
                return materialized(q, view, kwargs["qo_indptr"], manager._positions, is_prefill=True)
            return reference_attention(q, view, kwargs["qo_indptr"], manager._positions)
        def decode_reference(q, view, **kwargs):
            cu = torch.arange(q.shape[0] + 1, device=q.device)
            if materialized:
                return materialized(q, view, cu, manager._positions, is_prefill=False)
            return reference_attention(q, view, cu, manager._positions)
        owner.prefill_op.run = prefill_reference
        owner.decode_op.run = decode_reference
    selection_reference = (json.loads((Path(args.selection_reference) / "selections.json").read_text())
                           if args.selection_reference else None)
    if not args.token_layout:
        original_score = manager.score_chain_layer
        def checked_score(seq_id, layer):
            actual = original_score(seq_id, layer)
            if args.profile:
                return actual
            if args.budget_policy == "adakv":
                selection_scores[layer] = actual.cpu().numpy()
            expected = reference_scores(manager, seq_id, layer)
            error = 0.0
            for head, expected_head in enumerate(expected):
                candidate = actual[head, :expected_head.numel()]
                torch.testing.assert_close(candidate, expected_head, atol=2e-5, rtol=2e-4)
                error = max(error, float((candidate - expected_head).abs().max()))
            score_checks.append(dict(round=current_round, chain=seq_labels[seq_id], layer=layer, max_abs_error=error))
            return actual
        manager.score_chain_layer = checked_score
        original_selector = manager.chain_selector
        def checked_selector(seq_id):
            with profiler.record("head_slot_selection"):
                selected = original_selector(seq_id)
            row = manager.allocator.seq_rows[seq_id]
            key = f"{current_round}:{seq_labels[seq_id]}"
            if args.budget_policy == "adakv":
                occurrence = selection_counts.get(key, 0)
                selection_counts[key] = occurrence + 1
                key += f":{occurrence}"
            if args.budget_policy == "adakv" and not args.profile:
                from head_slot_reference import adakv_selection
                for layer, selection in enumerate(selected):
                    expected = adakv_selection(
                        selection_scores[layer], manager.allocator.lengths[layer, row],
                        budget=average_budget, alpha=args.adakv_alpha, sink=2, recent=4,
                        pool=config["pool_kernel_size"])
                    actual = [idx.cpu().tolist() for idx in selection.indices]
                    assert actual == expected, f"AdaKV selection differs at {key}, layer {layer}"
                    allocation_checks.append(dict(key=key, layer=layer, status="success",
                                                  head_budgets=list(map(len, actual))))
            if selection_reference is not None:
                from sparseengine.engine.cache_manager.base import HeadSlotSelection
                forced = []
                for l, heads in enumerate(selection_reference[key]):
                    indices = []
                    for h, retained_positions in enumerate(heads):
                        n = manager.allocator.lengths[l, row, h]
                        candidates = manager.allocator.positions[l, row, h, :n]
                        lookup = {int(pos): i for i, pos in enumerate(candidates)}
                        indices.append(torch.tensor([lookup[pos] for pos in retained_positions], device=manager.device))
                    forced.append(HeadSlotSelection(tuple(indices)))
                selected = forced
            selections[key] = [[manager.original_positions[l, row, h, idx].cpu().tolist()
                                for h, idx in enumerate(sel.indices)] for l, sel in enumerate(selected)]
            return selected
        manager.chain_selector = checked_selector

    original_run = llm.model_runner.run
    active_label = ""
    def observed_run(seqs, is_prefill):
        positions = [s.num_prefilled_tokens + s.current_chunk_size - 1 if is_prefill
                     else s.decode_input_position for s in seqs]
        with profiler.record("model_prefill" if is_prefill else "model_decode"):
            result = original_run(seqs, is_prefill)
        captured = getattr(llm.model_runner, "debug_last_logits", None)
        if captured is None:
            raise RuntimeError("Real model did not expose logits for validation.")
        if not args.profile:
            logits.append(captured[:len(seqs)].cpu())
        trace.append(dict(round=current_round, label=active_label, prefill=is_prefill,
                          chains=[seq_labels[s.seq_id] for s in seqs], positions=positions,
                          shape=list(captured[:len(seqs)].shape), sampled=result[0]))
        return result
    llm.model_runner.run = observed_run

    def drain(label):
        nonlocal active_label
        active_label = label
        for _ in range(config["max_model_len"] * 2):
            if llm.is_finished():
                return
            llm.step()
        raise RuntimeError("Bounded chain run did not finish.")

    def snapshot():
        if args.token_layout:
            return dict(free=manager.num_free_slots, chains=len(coordinator.index.records))
        return check_state(manager, coordinator)
    chain_ids = [None, None]
    try:
        for round_idx in range(3):
            current_round = round_idx
            admitted = []
            started = time.perf_counter()
            # Two interleaved chains use different prompts and continuation lengths.
            for b in range(2):
                if round_idx == 0:
                    text = ("Explain why the sky is blue. " if b == 0 else "Count the prime numbers below thirty. ")
                    base = llm.tokenizer.encode(text, add_special_tokens=True)
                    tokens = (base * (args.prompt_length // len(base) + 2))[:args.prompt_length - 7 * b]
                else:
                    tokens = llm.tokenizer.encode(" Continue briefly." if b == 0 else " Give another example.",
                                                   add_special_tokens=False) * (round_idx + b + 1)
                admission = llm.admit_request(tokens, params, chain_id=chain_ids[b],
                                              chain_append_only=round_idx > 0)
                requests.append(dict(round=round_idx, chain=b, token_ids=tokens,
                                     append_only=round_idx > 0, max_tokens=params.max_tokens,
                                     temperature=params.temperature, ignore_eos=params.ignore_eos))
                chain_ids[b] = admission.chain_id
                admitted.append(admission)
                seq_labels[admission.seq_id] = b
                if round_idx == 1:
                    drain(f"single-{b}")
            drain("paired")
            torch.cuda.synchronize()
            state = snapshot()
            assert all(coordinator.index.lookup(c).state is ChainState.IDLE for c in chain_ids)
            for a in admitted:
                if not args.no_compression:
                    row = manager.allocator.seq_rows[a.seq_id]
                    if args.budget_policy == "adakv":
                        before = manager.last_compaction[a.seq_id]["before"]
                        targets = np.minimum(before, hf.num_key_value_heads * average_budget)
                        np.testing.assert_array_equal(manager.allocator.lengths[:, row].sum(axis=1), targets)
                    else:
                        assert np.array_equal(manager.allocator.lengths[:, row], np.asarray(budgets))
            records.append(dict(round=round_idx, elapsed_s=time.perf_counter() - started,
                                state=state, compactions=dict(getattr(manager, "last_compaction", {}))))
            (output / "rounds.json").write_text(json.dumps(records, indent=2, default=str))
        (output / "selections.json").write_text(json.dumps(selections))
        (output / "requests.json").write_text(json.dumps(requests))
        (output / "score_checks.json").write_text(json.dumps(score_checks))
        (output / "allocation_checks.json").write_text(json.dumps(allocation_checks))
        adaptive_budgets = [[list(map(len, heads)) for heads in layers] for layers in selections.values()]
        nonuniform_layers = sum(len(set(heads)) > 1 for layers in adaptive_budgets for heads in layers)
        if args.budget_policy == "adakv" and args.adakv_alpha < 1:
            assert nonuniform_layers > 0, "Probe did not exercise adaptive nonuniform budgets"
        different_equal_budget_heads = 0
        if not args.no_compression:
            for layers in selections.values():
                for l, heads in enumerate(layers):
                    different_equal_budget_heads += sum(
                        budgets[l][a] == budgets[l][b] and heads[a] != heads[b]
                        for a in range(len(heads)) for b in range(a + 1, len(heads)))
            assert different_equal_budget_heads > 0, "Equal budgets must still allow independent selection"
        # Both rows are IDLE. A third chain must evict the oldest and reuse its row.
        current_round = 3
        oldest = min(coordinator.index.records.values(), key=lambda r: r.last_access)
        oldest_id = oldest.chain_id
        old_row = manager.allocator.seq_rows[oldest.seq_id] if not args.token_layout else None
        survivor_id = next(c for c in chain_ids if c != oldest_id)
        survivor = coordinator.index.lookup(survivor_id)
        if not args.token_layout:
            survivor_row = manager.allocator.seq_rows[survivor.seq_id]
            old_survivor = manager.slot_table[:, survivor_row].clone()
        # Stop forcing recorded selections for the extra lifecycle probe.
        selection_reference = None
        extra = llm.admit_request(llm.tokenizer.encode("Explain two benefits of clean water."), params)
        seq_labels[extra.seq_id] = 2
        assert oldest_id not in coordinator.index.records
        drain("lru-reuse")
        if not args.token_layout:
            assert manager.allocator.seq_rows[extra.seq_id] == old_row
            torch.testing.assert_close(manager.slot_table[:, survivor_row], old_survivor, atol=0, rtol=0)
        records.append(dict(event="idle_lru_row_reuse", state=snapshot()))
        llm.invalidate_chain(survivor_id)
        records.append(dict(event="delete", state=snapshot()))
        # Cancel before any forward: the reservation and prospective row return once.
        cancelled = llm.admit_request([100, 200, 300], params)
        seq_labels[cancelled.seq_id] = 3
        llm.abort_request(cancelled.seq_id)
        records.append(dict(event="cancel_unallocated", state=snapshot()))
        llm.invalidate_chain(extra.chain_id)
        final = snapshot()
        if not args.token_layout:
            assert not manager.allocator.seq_rows
            assert tuple(manager.allocator.free_counts) == manager.allocator.capacities
        counters_after = {name: getattr(graph, name, 0) for name in counter_names}
        delta = {k: counters_after[k] - counters_before[k] for k in counter_names}
        if not args.eager:
            assert counters_after["capture_count"] >= 2 and delta["replay_count"] > 0
            assert delta["capture_count"] == delta["recapture_count"] == delta["force_eager_count"] == 0
            assert {str(k): v.decode_state.inputs.data_ptrs() for k, v in graph._graphs.items()} == graph_ptrs
        summary = dict(status="success", rounds=3, final=final,
                       attention_binding=owner.binding_metadata(),
                       materialized_binding=(dict(prefill_provider=materialized.prefill.name,
                                                  decode_provider=materialized.decode.name)
                                             if materialized else None),
                       graph_plan=graph.graph_plan() if graph else None,
                       graph_counters=counters_after, measured_graph_counters=delta,
                       scorer_max_abs_error=max((x["max_abs_error"] for x in score_checks), default=None),
                       logits_tolerance=dict(atol=0.25, rtol=0.02),
                       different_equal_budget_heads=different_equal_budget_heads,
                       nonuniform_layers=nonuniform_layers,
                       actual_head_budgets=adaptive_budgets,
                       records=records)
        if args.check_materialized_attention:
            assert attention_checks, "Attention check mode did not observe any operation"
            summary["attention_oracle"] = dict(checks=len(attention_checks), atol=0.02, rtol=0.02,
                actual_fp32_max_abs_error=max(x["actual_fp32_max_abs_error"] for x in attention_checks),
                materialized_fp32_max_abs_error=max(x["materialized_fp32_max_abs_error"] for x in attention_checks),
                actual_materialized_max_abs_error=max(x["actual_materialized_max_abs_error"] for x in attention_checks))
        if args.profile:
            summary["profile"] = dict(
                timing="Synchronized component wall time; nested scopes overlap; includes first-call/JIT costs",
                scope="Three chain rounds plus LRU probe; oracle and extra logit copies disabled",
                components=profiler.snapshot(),
                physical_pool_bytes=manager.kv_cache.numel() * manager.kv_cache.element_size(),
                auxiliary_reserved_bytes=manager.auxiliary_reserved_bytes,
                retained_kv_bytes_per_round=[sum(r["state"]["allocated"]) *
                                            manager.attention_cache_bytes_per_slot_per_layer()
                                            for r in records if "round" in r])
            (output / "profile_samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in profile_samples))
        (output / "trace.json").write_text(json.dumps(trace, indent=2, default=str))
        torch.save(logits, output / "logits.pt")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps({k: v for k, v in summary.items() if k != "records"}, default=str))
    finally:
        if args.check_materialized_attention:
            (output / "attention_checks.json").write_text(json.dumps(attention_checks))
        profiler.record = saved_record
        profiler.set_enabled(False)
        if materialized:
            materialized.close()
        llm.exit()


if __name__ == "__main__":
    main()
