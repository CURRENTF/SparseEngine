"""Model runner debug snapshots and tensor-parallel replica diagnostics."""

import torch
import torch.distributed as dist

from sparseengine.engine.cache_manager.base import _debug_tensor_summary


def debug_sparse_state_summary(runner) -> dict[str, object]:
    def parallel_group_summary(group) -> dict[str, object] | None:
        if group is None:
            return None
        return {
            "rank": int(group.rank),
            "size": int(group.size),
            "ranks": [int(rank) for rank in group.ranks],
        }

    parallel_context = runner.parallel_context
    config = getattr(runner, "config", None)
    moe_synced = {}
    moe_local = {}
    model = getattr(getattr(runner, "model", None), "model", None)
    layers = getattr(model, "layers", ())
    selected_layers = {0, len(layers) - 1} if layers else set()
    for layer_idx, layer in enumerate(layers):
        if layer_idx not in selected_layers:
            continue
        block = getattr(layer, "mlp", None)
        topk_ids = getattr(block, "debug_last_topk_ids", None)
        topk_weights = getattr(block, "debug_last_topk_weights", None)
        output = getattr(block, "debug_last_output", None)
        if topk_ids is None or topk_weights is None or output is None:
            continue
        moe_synced[str(layer_idx)] = {
            "topk_ids": _debug_tensor_summary(topk_ids),
            "topk_weights": _debug_tensor_summary(topk_weights),
            "output": _debug_tensor_summary(output),
        }
        experts = block.experts
        moe_local[str(layer_idx)] = {
            "local_expert_start": int(experts.local_expert_start),
            "local_expert_end": int(experts.local_expert_end),
            "local_hit_count": int(block.debug_last_local_hit_count),
            "local_output": _debug_tensor_summary(block.debug_last_local_output),
        }
    state = runner.sparse_controller.debug_state_summary()
    prefix_cache_coordinator = getattr(
        runner,
        "prefix_cache_coordinator",
        None,
    )
    if prefix_cache_coordinator is not None:
        state["mixed_prefix_cache"] = (
            prefix_cache_coordinator.debug_state_summary()
        )
    graph_runner = getattr(runner, "decode_graph_runner", None)
    graph_key = getattr(graph_runner, "last_state_key", None)
    graph_summary = {
        "dp_idle_graph_count": len(getattr(runner, "dp_idle_graphs", {})),
        "dp_idle_replay_count": int(getattr(runner, "dp_idle_replay_count", 0)),
        "dp_idle_eager_count": int(getattr(runner, "dp_idle_eager_count", 0)),
        "enabled": bool(
            getattr(config, "decode_graph", False)
        ),
        "capture_count": int(
            getattr(graph_runner, "capture_count", 0)
        ),
        "replay_count": int(
            getattr(graph_runner, "replay_count", 0)
        ),
        "eager_static_count": int(
            getattr(graph_runner, "eager_static_count", 0)
        ),
        "force_eager_count": int(
            getattr(graph_runner, "force_eager_count", 0)
        ),
        "eviction_count": int(
            getattr(graph_runner, "eviction_count", 0)
        ),
        "recapture_count": int(
            getattr(graph_runner, "recapture_count", 0)
        ),
        "cached_graph_count": len(
            getattr(graph_runner, "_graphs", {})
        ),
        "graph_plan": (
            graph_runner.graph_plan()
            if callable(getattr(graph_runner, "graph_plan", None))
            else None
        ),
        "last_state_key": (
            {
                "method": str(graph_key.method or ""),
                "batch_size": int(graph_key.batch_size),
                "graph_path_id": str(graph_key.graph_path_id),
                "capture_sampling": bool(graph_key.capture_sampling),
            }
            if graph_key is not None
            else None
        ),
    }
    return {
        "world_rank": runner.parallel_context.world_rank,
        "ep_rank": runner.parallel_context.moe_ep_rank,
        "parallel": {
            "configured": {
                "tensor_parallel_size": int(
                    getattr(config, "tensor_parallel_size", parallel_context.attn_tp_size)
                ),
                "expert_parallel_size": int(
                    getattr(config, "expert_parallel_size", parallel_context.moe_ep_size)
                ),
                "data_parallel_size": int(
                    getattr(config, "data_parallel_size", parallel_context.attn_dp_size)
                ),
                "world_size": int(
                    getattr(config, "world_size", parallel_context.world_size)
                ),
            },
            "effective": {
                "world": parallel_group_summary(parallel_context.world),
                "attn_tp": parallel_group_summary(parallel_context.attn_tp),
                "moe_ep": parallel_group_summary(parallel_context.moe_ep),
                "moe_tp": parallel_group_summary(
                    parallel_context.moe_tp
                ),
                "attn_dp": parallel_group_summary(parallel_context.attn_dp),
            },
        },
        "state": state,
        "decode_graph": graph_summary,
        "last_logits": (
            _debug_tensor_summary(runner.debug_last_logits)
            if hasattr(runner, "debug_last_logits")
            else None
        ),
        "moe_synced": moe_synced,
        "moe_local": moe_local,
    }


def debug_last_logits_cpu(runner) -> torch.Tensor | None:
    if runner.parallel_context.attn_tp_rank != 0:
        return None
    logits = getattr(runner, "debug_last_logits", None)
    if logits is None:
        raise RuntimeError(
            "No debug logits are available. Set SPARSEENGINE_DEBUG_RUNTIME=1 before engine startup."
        )
    return logits.detach().cpu()


def debug_hidden_states_cpu(runner) -> dict[int, torch.Tensor] | None:
    model = getattr(getattr(runner, "model", None), "model", None)
    snapshots = getattr(model, "debug_last_hidden_states", None)
    if snapshots is None:
        raise RuntimeError(
            "No hidden-state snapshots are available. Set "
            "SPARSEENGINE_DEBUG_HIDDEN_LAYERS before model execution."
        )
    if runner.parallel_context.attn_tp_rank != 0:
        return None
    return {
        int(layer_idx): tensor.detach().cpu()
        for layer_idx, tensor in snapshots.items()
    }


def debug_moe_states_cpu(runner) -> dict[int, dict[str, torch.Tensor]] | None:
    model = getattr(getattr(runner, "model", None), "model", None)
    layers = getattr(model, "layers", ())
    snapshots = {}
    for layer_idx, layer in enumerate(layers):
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "experts"):
            continue
        required = {
            "input": getattr(block, "debug_last_input", None),
            "topk_ids": getattr(block, "debug_last_topk_ids", None),
            "topk_weights": getattr(block, "debug_last_topk_weights", None),
            "output": getattr(block, "debug_last_output", None),
        }
        missing = [name for name, tensor in required.items() if tensor is None]
        if missing:
            raise RuntimeError(
                f"Layer {layer_idx} is missing MoE debug tensors {missing}. Set "
                "SPARSEENGINE_DEBUG_MOE before model execution."
            )
        snapshots[layer_idx] = {
            name: tensor.detach().cpu()
            for name, tensor in required.items()
        }
    return snapshots if runner.parallel_context.attn_tp_rank == 0 else None


def _debug_float_error_from_tp_leader(
    runner,
    tensor: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    if runner.parallel_context.attn_tp_size == 1:
        return 0.0, 0.0
    reference = tensor.detach().clone()
    dist.broadcast(
        reference,
        src=runner.parallel_context.attn_tp.ranks[0],
        group=runner.parallel_context.attn_tp.process_group,
    )
    difference = (tensor.detach().float() - reference.float()).abs()
    max_abs = difference.max()
    tolerance_ratio = (
        difference / (float(atol) + float(rtol) * reference.float().abs())
    ).max()
    runner.parallel_context.attn_tp.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    runner.parallel_context.attn_tp.all_reduce(tolerance_ratio, op=dist.ReduceOp.MAX)
    return float(max_abs.item()), float(tolerance_ratio.item())


def _debug_any_mismatch_from_tp_leader(runner, tensor: torch.Tensor) -> bool:
    if runner.parallel_context.attn_tp_size == 1:
        return False
    reference = tensor.detach().clone()
    dist.broadcast(
        reference,
        src=runner.parallel_context.attn_tp.ranks[0],
        group=runner.parallel_context.attn_tp.process_group,
    )
    mismatch = torch.tensor(
        [int(not torch.equal(tensor.detach(), reference))],
        dtype=torch.int32,
        device=runner.device,
    )
    runner.parallel_context.attn_tp.all_reduce(mismatch, op=dist.ReduceOp.MAX)
    return bool(mismatch.item())


def debug_replica_consistency(runner) -> dict[str, object] | None:
    logits = getattr(runner, "debug_last_logits", None)
    if runner.parallel_context.attn_tp_size > 1:
        result: dict[str, object] = {
            "last_logits_max_abs": None,
            "last_logits_tolerance_ratio": None,
            "last_logits_comparison": "not_applicable_tp_vocab_sharded",
            "moe_layers": {},
        }
    else:
        if logits is None:
            return None
        logits_max_abs, logits_tolerance_ratio = (
            runner._debug_float_error_from_tp_leader(
                logits,
                atol=0.05,
                rtol=0.05,
            )
        )
        result = {
            "last_logits_max_abs": logits_max_abs,
            "last_logits_tolerance_ratio": logits_tolerance_ratio,
            "last_logits_comparison": "compared",
            "moe_layers": {},
        }
    model = getattr(getattr(runner, "model", None), "model", None)
    layers = getattr(model, "layers", ())
    for layer_idx in sorted({0, len(layers) - 1} if layers else set()):
        block = getattr(layers[layer_idx], "mlp", None)
        if not hasattr(block, "debug_last_topk_ids"):
            continue
        topk_weights_max_abs, topk_weights_tolerance_ratio = (
            runner._debug_float_error_from_tp_leader(
                block.debug_last_topk_weights,
                atol=0.01,
                rtol=0.01,
            )
        )
        output_max_abs, output_tolerance_ratio = (
            runner._debug_float_error_from_tp_leader(
                block.debug_last_output,
                atol=0.05,
                rtol=0.05,
            )
        )
        result["moe_layers"][str(layer_idx)] = {
            "topk_ids_mismatch": runner._debug_any_mismatch_from_tp_leader(
                block.debug_last_topk_ids
            ),
            "topk_weights_max_abs": topk_weights_max_abs,
            "topk_weights_tolerance_ratio": topk_weights_tolerance_ratio,
            "output_max_abs": output_max_abs,
            "output_tolerance_ratio": output_tolerance_ratio,
        }
    return result


def debug_sparse_state_summaries(runner, synchronize: bool = False) -> list[dict[str, object]] | None:
    local_error: BaseException | None = None
    local_summary = None
    try:
        if synchronize:
            runner.platform.synchronize()
        local_summary = runner.debug_sparse_state_summary()
        local_summary["replica_consistency"] = (
            runner.debug_replica_consistency()
        )
    except BaseException as exc:
        local_error = exc
    runner._sync_tp_rpc_status("debug_sparse_state_summaries", local_error)
    if local_error is not None:
        raise local_error
    if runner.parallel_context.attn_tp_size == 1:
        return [local_summary]
    summaries = [None] * runner.parallel_context.attn_tp_size
    dist.all_gather_object(
        summaries,
        local_summary,
        group=runner.parallel_context.attn_tp.process_group,
    )
    return summaries if runner.parallel_context.attn_tp_rank == 0 else None
