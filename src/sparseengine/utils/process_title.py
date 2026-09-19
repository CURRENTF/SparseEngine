from sparseengine.distributed.parallel_context import ParallelContext


def engine_process_title(parallel_context: ParallelContext) -> str:
    if parallel_context.world_size == 1:
        return "SENGINE_Engine"

    title = "SENGINE"
    if parallel_context.attn_tp_size > 1:
        title += f"_TP{parallel_context.attn_tp_rank}"
    if parallel_context.attn_dp_size > 1:
        title += f"_DP{parallel_context.attn_dp_rank}"
    if parallel_context.moe_ep_size > 1:
        title += f"_EP{parallel_context.moe_ep_rank}"
    return title


def set_engine_process_title(parallel_context: ParallelContext) -> None:
    import setproctitle

    setproctitle.setproctitle(engine_process_title(parallel_context))
