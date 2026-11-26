import math
import pickle
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparseengine.distributed.topology import ParallelTopology
from sparseengine.engine.cache_manager.base import (
    AttentionViewMeta,
    ExplicitKVPayload,
    PrefillComputeView,
)
from sparseengine.engine.cache_manager.methods.h2o import H2OCacheManager
from sparseengine.engine.cache_manager.methods.pace import PACECacheManager
from sparseengine.engine.pace_plan import PACEPlan, compile_bonus
from sparseengine.engine.sequence import Sequence
from sparseengine.method_registry import (
    resolve_cache_sparse_method,
    validate_model_runtime_compatibility,
)
from sparseengine.utils.context import reset_context, set_context

from test_h2o_cache_manager import _manager_with_layer_rows

SCALE = 1e5
DECAY = 0.35


@pytest.fixture(autouse=True)
def _reset_runtime_context():
    reset_context()
    yield
    reset_context()


def _oracle_bonus(positions, spans, step, *, scale=SCALE, decay=DECAY):
    """Per-span reference: max over covering spans of 0.5 + 0.5 exp(-decay * gap)."""
    out = torch.zeros(len(positions), dtype=torch.float32)
    for start, end, steps in spans:
        upcoming = [s for s in sorted(set(steps)) if s >= step]
        if not upcoming:
            continue
        weight = 0.5 + 0.5 * math.exp(-decay * float(upcoming[0] - step))
        covered = torch.tensor([max(0, start) <= p < end for p in positions])
        out = torch.where(covered, torch.maximum(out, torch.full_like(out, weight)), out)
    return out * float(scale)


def test_pace_compiled_bonus_matches_per_span_oracle():
    rng = random.Random(7)
    for _ in range(300):
        spans = []
        for _ in range(rng.randint(0, 12)):
            start = rng.randint(-5, 120)
            spans.append((start, start + rng.randint(-3, 40),
                          [rng.randint(0, 30) for _ in range(rng.randint(0, 4))]))
        step = rng.randint(0, 30)
        plan = PACEPlan(step, tuple(spans))
        positions = list(range(0, 170))
        got = compile_bonus(plan, scale=SCALE, decay=DECAY).lookup(
            torch.tensor(positions, dtype=torch.int64)
        )
        assert torch.equal(got, _oracle_bonus(positions, spans, step))


def test_pace_plan_normalization_makes_equal_plans_compare_equal():
    a = PACEPlan(3, ((5, 9, [7, 4, 7]), (4, 4, [1]), (-2, 3, ())))
    b = PACEPlan(3, ((5, 9, (4, 7)), (0, 3, ())))
    assert a == b
    assert a.spans == ((0, 3, ()), (5, 9, (4, 7)))
    with pytest.raises(TypeError):
        PACEPlan(True)
    with pytest.raises(ValueError):
        PACEPlan(0, ((0, 4, [-1]),))


def test_pace_plan_survives_sequence_ipc():
    plan = PACEPlan(2, ((0, 4, (3,)),))
    seq = Sequence(list(range(8)), pace_plan=plan)
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.pace_plan == plan
    with pytest.raises(TypeError):
        Sequence([1, 2], pace_plan={"current_step": 0})


# ---------------------------------------------------------------------------
# Manager lifecycle on CPU
# ---------------------------------------------------------------------------
def _manager(cls, *, layers=2, decode_budget=4, prefill_budget=6, interval=1):
    manager = _manager_with_layer_rows(
        [[0] for _ in range(layers)],
        decode_budget=decode_budget,
        decode_eviction_interval=interval,
        prefill_budget=prefill_budget,
    )
    manager.__class__ = cls
    manager.config.sparse_method = "pace" if cls is PACECacheManager else "h2o"
    manager.config.pace_bonus_scale = SCALE
    manager.config.pace_bonus_decay = DECAY
    if cls is PACECacheManager:
        manager._pace_init_state()
    manager._next_slot = 1000
    return manager


def _tag(manager, layer, slots, logical):
    # V carries each token's logical position, so the oracle survives page
    # table compaction and dense final-prefill moves, which relocate K with V;
    # V is never re-rotated.
    _, v_cache = manager.get_layer_kv_cache(layer)
    v_cache[slots.long()] = logical.view(-1, 1, 1).to(v_cache.dtype)


def _resident_logical(manager, layer, seq_id=0):
    row = manager.seq_id_to_row[layer][seq_id]
    n = int(manager.row_seq_lens[layer][row])
    slots = manager.buffer_req_to_token_slots[layer][row, :n].long()
    _, v_cache = manager.get_layer_kv_cache(layer)
    return v_cache[slots, 0, 0].to(torch.int64)


def _append_columns(manager, layer, logical_start, count, seq_id=0):
    row = manager.seq_id_to_row[layer][seq_id]
    n = int(manager.row_seq_lens[layer][row])
    slots = torch.arange(manager._next_slot, manager._next_slot + count, dtype=torch.int32)
    manager._next_slot += count
    manager.buffer_req_to_token_slots[layer][row, n:n + count] = slots
    manager.row_seq_lens[layer][row] = n + count
    _tag(manager, layer, slots, torch.arange(logical_start, logical_start + count))
    return n


def _step_score(layer, logical_start, length, total):
    # Distinct, deterministic attention per column; no ties.
    base = torch.arange(total, dtype=torch.float32)
    return ((base * 0.37 + layer * 0.11 + logical_start * 0.013) % 1.0 + 0.01).unsqueeze(0)


def _prefill_chunk(manager, seq, chunk, *, bias=None):
    for layer in manager.kv_transformer_layer_indices():
        cached = _append_columns(manager, layer, seq.num_prefilled_tokens, chunk)
        total = cached + chunk
        scores = _step_score(layer, seq.num_prefilled_tokens, chunk, total)
        if bias is not None:
            scores[0, cached:] += bias(layer, seq.num_prefilled_tokens, chunk, cached)
        view = PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=manager.buffer_req_to_token_slots[layer],
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([total], dtype=torch.int32),
                max_context_len=total,
            ),
            payload=ExplicitKVPayload(
                k_cache=torch.empty((16, 1, 1)), v_cache=torch.empty((16, 1, 1))
            ),
            token_scores=scores,
        )
        set_context(is_prefill=True, cache_manager=manager, seqs=[seq])
        manager.collect_prefill_attention_score(
            layer,
            torch.empty((chunk, 1, 1)),
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([chunk], dtype=torch.int32),
        )
        reset_context()
    if seq.is_last_chunk_prefill:
        manager.compact_final_prefill_for_decode([seq])
    else:
        manager.evict_after_intermediate_prefill([seq])
    seq.num_prefilled_tokens += chunk


def _decode(manager, seq, logical, score):
    for layer in manager.kv_transformer_layer_indices():
        _append_columns(manager, layer, logical, 1)
        key = (layer, 0)
        manager._h2o_scores[key] = torch.cat(
            (manager._h2o_scores[key], torch.tensor([score + layer], dtype=torch.float32))
        )
    manager.evict_after_decode([seq])


def _run(manager, plan, *, prompt=12, chunk=4, decode_steps=5, bias=None):
    seq = Sequence(list(range(prompt)), pace_plan=plan)
    seq.seq_id = 0
    while seq.num_prefilled_tokens < prompt:
        seq.current_chunk_size = chunk
        _prefill_chunk(manager, seq, chunk, bias=bias)
    for i in range(decode_steps):
        _decode(manager, seq, prompt + i, 0.05 * (i % 3))
    return seq


def test_pace_without_plan_retains_and_scores_exactly_like_h2o():
    h2o, pace = _manager(H2OCacheManager), _manager(PACECacheManager)
    _run(h2o, None)
    _run(pace, None)
    for layer in h2o.kv_transformer_layer_indices():
        assert torch.equal(_resident_logical(pace, layer), _resident_logical(h2o, layer))
        assert torch.equal(pace._h2o_scores[(layer, 0)], h2o._h2o_scores[(layer, 0)])


def test_decode_steps_record_row_state_lazily():
    """A decode step that compacts nothing does no PACE bookkeeping; readers still
    see every column."""
    plan = PACEPlan(1, ((1, 3, (2,)), (6, 8, (5, 9))))
    pace = _manager(PACECacheManager, interval=8)
    seq = _run(pace, plan, decode_steps=0)
    layers = list(pace.kv_transformer_layer_indices())
    quiet = compacting = 0
    for i in range(20):
        recorded = {layer: pace._pace_recorded((layer, 0)) for layer in layers}
        evictions = pace._h2o_counters["decode_evictions"]
        _decode(pace, seq, 12 + i, 0.05 * (i % 3))
        if pace._h2o_counters["decode_evictions"] == evictions:
            quiet += 1
            for layer in layers:
                assert pace._pace_recorded((layer, 0)) == recorded[layer]
        else:
            compacting += 1
    assert quiet and compacting
    for layer in layers:
        state = pace.pace_row_state(layer, 0)
        assert torch.equal(state["logical"], _resident_logical(pace, layer))
        assert all(int(state[name].numel()) == int(state["logical"].numel())
                   for name in ("bonus", "physical", "ref"))


def test_pace_slot_cost_includes_the_row_aligned_state():
    from sparseengine.engine.cache_manager.methods.pace import pace_row_state_bytes_per_slot
    config = SimpleNamespace(sparse_method="pace")
    # logical, physical and ref in int64, bonus in float32, per slot and layer
    assert pace_row_state_bytes_per_slot(config) == 3 * 8 + 4
    config.sparse_method = "h2o"
    assert pace_row_state_bytes_per_slot(config) == 0
    pace = _manager(PACECacheManager)
    with patch.object(H2OCacheManager, "_get_available_slots_info", return_value=(10_000, 100)):
        assert pace._get_available_slots_info()[1] == 100 + 28


def test_pace_row_state_tracks_logical_positions_through_eviction():
    plan = PACEPlan(1, ((1, 3, (2,)), (6, 8, (5, 9))))
    pace = _manager(PACECacheManager)
    _run(pace, plan)
    for layer in pace.kv_transformer_layer_indices():
        state = pace.pace_row_state(layer, 0)
        logical, bonus = state["logical"], state["bonus"]
        resident = _resident_logical(pace, layer)
        assert torch.equal(logical, resident)
        expected = _oracle_bonus(resident.tolist(), plan.spans, plan.current_step)
        decode = resident >= 12
        expected[decode] = 0.0  # decode tokens carry no bonus at creation
        assert torch.equal(bonus, expected)


def test_pace_equals_h2o_over_foresight_adjusted_scores():
    plan = PACEPlan(1, ((1, 3, (2,)), (6, 8, (5, 9))))
    pace, h2o = _manager(PACECacheManager), _manager(H2OCacheManager)
    _run(pace, plan)

    def bias(layer, start, chunk, cached):
        # H2O weights a chunk's step score by its scored query count.
        queries = min(chunk, int(h2o.config.h2o_prefill_score_window) or chunk)
        return _oracle_bonus(range(start, start + chunk), plan.spans, 1) / queries

    _run(h2o, None, bias=bias)
    for layer in pace.kv_transformer_layer_indices():
        assert torch.equal(_resident_logical(pace, layer), _resident_logical(h2o, layer))
        torch.testing.assert_close(
            pace._h2o_scores[(layer, 0)], h2o._h2o_scores[(layer, 0)], rtol=1e-6, atol=1e-2
        )
    # Budget 4 with recent_ratio 0.5 leaves two heavy-hitter places: the span
    # needed soonest (next reference at step 2) wins over the later one.
    for layer in pace.kv_transformer_layer_indices():
        resident = set(_resident_logical(pace, layer).tolist())
        assert {1, 2} <= resident and not {6, 7} & resident


def test_pace_turn_rebase_replaces_folded_bonus_with_new_plan():
    first = PACEPlan(1, ((1, 3, (2,)), (6, 8, (5,))))
    pace = _manager(PACECacheManager, decode_budget=6, prefill_budget=8)
    seq = _run(pace, first, decode_steps=2)
    processed = 12 + 2
    pace.on_chain_turn_finished(0, processed_token_count=processed)
    before = {layer: pace._h2o_scores[(layer, 0)].clone()
              for layer in pace.kv_transformer_layer_indices()}

    second = PACEPlan(6, ((1, 3, (2,)), (6, 8, (6, 9)), (12, 14, (7,))))
    turn = Sequence(list(range(processed + 4)), pace_plan=second)
    turn.seq_id = 0
    turn.chain_reused_tokens = turn.num_prefilled_tokens = processed
    turn.current_chunk_size = 4
    with patch.object(H2OCacheManager, "prepare_step", return_value=None):
        pace.prepare_step([turn], True)

    for layer in pace.kv_transformer_layer_indices():
        state = pace.pace_row_state(layer, 0)
        logical, bonus = state["logical"], state["bonus"]
        first_bonus = _oracle_bonus(logical.tolist(), first.spans, 1)
        first_bonus[logical >= 12] = 0.0
        new_bonus = _oracle_bonus(logical.tolist(), second.spans, 6)
        assert torch.equal(bonus, new_bonus)
        torch.testing.assert_close(
            pace._h2o_scores[(layer, 0)], before[layer] - first_bonus + new_bonus
        )


def test_pace_chain_turn_end_rejects_lost_bookkeeping():
    pace = _manager(PACECacheManager)
    _run(pace, None, decode_steps=1)
    with pytest.raises(RuntimeError, match="processed_token_count"):
        pace.on_chain_turn_finished(0, processed_token_count=99)


def test_pace_prefill_chunk_must_continue_recorded_row():
    pace = _manager(PACECacheManager)
    seq = _run(pace, None, decode_steps=0)
    seq.num_prefilled_tokens = 20  # skips logical positions 12..19
    seq.token_ids = list(range(30))
    seq.num_tokens = seq.num_prompt_tokens = 30
    seq.current_chunk_size = 4
    with pytest.raises(RuntimeError, match="does not continue the recorded row"):
        _prefill_chunk(pace, seq, 4)


# ---------------------------------------------------------------------------
# Registration and configuration boundaries
# ---------------------------------------------------------------------------
def test_pace_owns_the_h2o_prefill_lifecycle():
    assert resolve_cache_sparse_method("pace", prefill_sparse_method=None) == "pace"
    assert resolve_cache_sparse_method("h2o", prefill_sparse_method=None) == "h2o"
    assert resolve_cache_sparse_method("", prefill_sparse_method="h2o_prefill") == "h2o"


@pytest.mark.parametrize(
    ("model_type", "topology", "message"),
    [
        ("qwen3_5", ParallelTopology(1, 1, 1), "requires one of"),
        ("gemma4", ParallelTopology(1, 1, 1), "requires one of"),
        ("qwen3", ParallelTopology(2, 1, 1), "TP=EP=DP=1"),
        ("qwen3", ParallelTopology(1, 1, 2), "TP=EP=DP=1"),
    ],
)
def test_pace_rejects_unvalidated_models_and_topologies(model_type, topology, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        validate_model_runtime_compatibility(
            model_type=model_type, sparse_method="pace", topology=topology,
            decode_graph=False, enable_prefix_caching=False,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"prefill_sparse_method": ""}, NotImplementedError, "h2o_prefill"),
        ({"kv_cache_layout": "head_slots"}, NotImplementedError, "kv_cache_layout"),
        ({"attention_cache_layout": "mla_latent"}, NotImplementedError, "explicit KV"),
        ({"enable_multimodal": True}, NotImplementedError, "multimodal"),
        ({"pace_bonus_scale": 0.0}, ValueError, "pace_bonus_scale"),
        ({"pace_bonus_decay": float("nan")}, ValueError, "pace_bonus_decay"),
    ],
)
def test_pace_config_rejects_unsupported_combinations(overrides, error, message):
    from sparseengine.configs.sparse import _normalize_pace

    config = SimpleNamespace(
        sparse_method="pace", prefill_sparse_method="h2o_prefill",
        attention_cache_layout="explicit_kv", kv_cache_layout="token",
        enable_multimodal=False, pace_bonus_scale=SCALE, pace_bonus_decay=DECAY,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    with pytest.raises(error, match=message):
        _normalize_pace(config)


def test_pace_plan_is_rejected_for_other_methods():
    from sparseengine.engine.llm_engine import LLMEngine

    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(resolved_cache_sparse_method="h2o")
    with pytest.raises(ValueError, match="requires sparse_method='pace'"):
        engine.admit_request([1, 2, 3], None, pace_plan=PACEPlan(0))


# ---------------------------------------------------------------------------
# Bounded positions
# ---------------------------------------------------------------------------
from sparseengine.engine.cache_manager.methods.pace_positions import (  # noqa: E402
    PACEReferenceStore,
    rotate_keys,
)
from sparseengine.layers.rotary_embedding import apply_rotary_emb  # noqa: E402


def _cos_sin(max_pos, head_dim, *, base=10000.0, scaling=1.0, dtype=torch.float32):
    # RotaryEmbedding's construction, optionally with YaRN-style magnitude.
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float64), inv_freq)
    return torch.cat((freqs.cos() * scaling, freqs.sin() * scaling), -1).unsqueeze(1).to(dtype)


def _rotate_at(raw, table, pos):
    cos, sin = table[pos.long()].float().chunk(2, dim=-1)
    return apply_rotary_emb(raw, cos, sin)


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA")),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scaling", [1.0, 1.21])
def test_pace_delta_rotation_equals_fresh_rotation(scaling, device):
    torch.manual_seed(0)
    table = _cos_sin(512, 64, scaling=scaling).to(device)
    raw = torch.randn(40, 4, 64, dtype=torch.float64, device=device)
    old = torch.randint(0, 512, (40,), device=device)
    new = torch.randint(0, 512, (40,), device=device)
    moved = rotate_keys(_rotate_at(raw, table.double(), old), table, old, new)
    torch.testing.assert_close(moved, _rotate_at(raw, table.double(), new), rtol=0, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_pace_reference_store_bounds_rounding_across_renumbers(device):
    torch.manual_seed(1)
    table = _cos_sin(4096, 128).to(device)
    raw = torch.randn(32, 2, 128, dtype=torch.float64, device=device)
    pos = torch.randint(2048, 4096, (32,), device=device)
    stored = {"reference": None, "in_place": None}
    errors = {}
    for mode in stored:
        k_cache = _rotate_at(raw, table.double(), pos).to(torch.bfloat16)
        slots = torch.arange(32, device=device)
        store = (PACEReferenceStore(capacity=32, num_kv_heads=2, head_dim=128,
                                    dtype=torch.bfloat16, device=torch.device(device))
                 if mode == "reference" else None)
        entries = torch.full((32,), -1, dtype=torch.int64, device=device)
        current = pos.clone()
        for step in range(200):
            target = (current * 7 + 13 * step) % 2048
            if store is None:
                k_cache.index_copy_(0, slots, rotate_keys(k_cache, table, current, target))
            else:
                entries, overflow = store.renumber(k_cache, slots, entries, current, target, table)
                assert overflow == 0
            current = target
        oracle = _rotate_at(raw, table.double(), current)
        errors[mode] = ((k_cache.double() - oracle).norm() / oracle.norm()).item()
    # bf16 has ~3e-3 relative precision per rounding: the reference copy stays at
    # a couple of roundings; repeated in-place rotation compounds.
    assert errors["reference"] < 1e-2
    assert errors["in_place"] > 3 * errors["reference"]


def _raw_key(logical, heads, dim):
    base = torch.arange(heads * dim, dtype=torch.float64).view(1, heads, dim)
    return torch.cos(logical.view(-1, 1, 1).double() * 0.37 + base) + 1.5


def _bounded_manager(limit=80, reference_tokens=8):
    pace = _manager(PACECacheManager)
    pace.config.pace_bounded_positions = True
    pace.config.pace_resolved_position_limit = limit
    pace.config.pace_reference_tokens = reference_tokens
    pace._pace_cos_sin = _cos_sin(128, pace.head_dim)
    pace.config.hf_config = SimpleNamespace(dtype=torch.float32, head_dim=pace.head_dim,
                                            model_type="qwen3")
    for layer in pace.kv_transformer_layer_indices() if reference_tokens else ():
        k_cache, _ = pace.get_layer_kv_cache(layer)
        pace._pace_stores[layer] = PACEReferenceStore(
            capacity=reference_tokens, num_kv_heads=k_cache.shape[-2],
            head_dim=k_cache.shape[-1], dtype=k_cache.dtype, device=k_cache.device,
        )
    return pace


def _write_keys(manager, layer, logical, physical):
    # The model: K is the raw key of the token rotated at its physical position.
    row = manager.seq_id_to_row[layer][0]
    n = int(manager.row_seq_lens[layer][row])
    slots = manager.buffer_req_to_token_slots[layer][row, n - len(logical):n].long()
    k_cache, _ = manager.get_layer_kv_cache(layer)
    raw = _raw_key(logical, k_cache.shape[-2], k_cache.shape[-1])
    k_cache[slots] = _rotate_at(raw, manager._pace_cos_sin.double(), physical).to(k_cache.dtype)


# 8: every resident key has a reference; 2: most overflow to in-place rotation;
# 0: no reference store.
@pytest.mark.parametrize("reference_tokens", [8, 2, 0])
def test_pace_bounded_session_keeps_positions_and_keys_consistent(reference_tokens):
    pace = _bounded_manager(reference_tokens=reference_tokens)
    prompt, chunk = 12, 4
    seq = Sequence(list(range(prompt)))
    seq.seq_id = 0
    max_physical = 0
    while seq.num_prefilled_tokens < prompt:
        seq.current_chunk_size = chunk
        pace._pace_apply_offsets([seq], True)
        start = seq.num_prefilled_tokens
        physical = torch.arange(start, start + chunk) - seq.position_offset
        max_physical = max(max_physical, int(physical.max()))
        for layer in pace.kv_transformer_layer_indices():
            _append_columns(pace, layer, start, chunk)
            _write_keys(pace, layer, torch.arange(start, start + chunk), physical)
        # Score the chunk like _prefill_chunk; its columns are already appended.
        for layer in pace.kv_transformer_layer_indices():
            row = pace.seq_id_to_row[layer][0]
            total = int(pace.row_seq_lens[layer][row])
            view = PrefillComputeView(
                meta=AttentionViewMeta(
                    active_slots=pace.buffer_req_to_token_slots[layer],
                    req_indices=torch.tensor([0], dtype=torch.int32),
                    context_lens=torch.tensor([total], dtype=torch.int32),
                    max_context_len=total,
                ),
                payload=ExplicitKVPayload(k_cache=torch.empty((16, 1, 1)),
                                          v_cache=torch.empty((16, 1, 1))),
                token_scores=_step_score(layer, start, chunk, total),
            )
            set_context(is_prefill=True, cache_manager=pace, seqs=[seq])
            pace.collect_prefill_attention_score(
                layer, torch.empty((chunk, 1, 1)), view,
                b_start_loc=torch.tensor([0], dtype=torch.int32),
                chunk_lens=torch.tensor([chunk], dtype=torch.int32),
            )
            reset_context()
        if seq.is_last_chunk_prefill:
            pace.compact_final_prefill_for_decode([seq])
        else:
            pace.evict_after_intermediate_prefill([seq])
        seq.num_prefilled_tokens += chunk

    for i in range(200):
        logical = prompt + i
        seq.token_ids.append(logical)
        seq.num_tokens = logical + 1
        pace._pace_apply_offsets([seq], False)
        physical = int(seq.decode_input_position)
        assert physical == logical - pace.pace_shift(0)
        max_physical = max(max_physical, physical)
        for layer in pace.kv_transformer_layer_indices():
            _append_columns(pace, layer, logical, 1)
            _write_keys(pace, layer, torch.tensor([logical]), torch.tensor([physical]))
            key = (layer, 0)
            pace._h2o_scores[key] = torch.cat(
                (pace._h2o_scores[key], torch.tensor([0.05 * (i % 3)], dtype=torch.float32))
            )
        pace.evict_after_decode([seq])

    assert pace._pace_counters["renumbers"] > 10
    assert max_physical < pace._pace_limit
    for layer in pace.kv_transformer_layer_indices():
        state = pace.pace_row_state(layer, 0)
        resident = _resident_logical(pace, layer)
        assert torch.equal(state["logical"], resident)
        assert bool((state["physical"][1:] > state["physical"][:-1]).all())
        row = pace.seq_id_to_row[layer][0]
        n = int(pace.row_seq_lens[layer][row])
        slots = pace.buffer_req_to_token_slots[layer][row, :n].long()
        k_cache, _ = pace.get_layer_kv_cache(layer)
        oracle = _rotate_at(_raw_key(resident, k_cache.shape[-2], k_cache.shape[-1]),
                            pace._pace_cos_sin.double(), state["physical"])
        torch.testing.assert_close(k_cache[slots].double(), oracle, rtol=0, atol=1e-4)
        # Only resident columns own reference entries.
        live = state["ref"][state["ref"] >= 0]
        if reference_tokens:
            assert int(pace._pace_stores[layer].in_use.sum()) == int(live.numel())
            assert len(set(live.tolist())) == int(live.numel())
        else:
            assert live.numel() == 0
    if reference_tokens == 2:
        assert pace._pace_counters["reference_overflow"] > 0


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"h2o_decode_eviction": False}, ValueError, "h2o_decode_eviction"),
        ({"decode_graph": True}, NotImplementedError, "decode_graph=False"),
        ({"pace_position_limit": 64}, ValueError, "too small"),
        ({"max_model_len": 32}, ValueError, "physical row capacity"),
        ({"pace_bounded_positions": False, "pace_position_limit": 4096}, ValueError,
         "requires pace_bounded_positions"),
    ],
)
def test_pace_bounded_config_rejects_unsound_settings(overrides, error, message):
    from sparseengine.configs.sparse import _normalize_pace

    config = SimpleNamespace(
        sparse_method="pace", prefill_sparse_method="h2o_prefill",
        attention_cache_layout="explicit_kv", kv_cache_layout="token",
        enable_multimodal=False, pace_bonus_scale=SCALE, pace_bonus_decay=DECAY,
        pace_bounded_positions=True, pace_position_limit=None, pace_reference_tokens=16,
        h2o_decode_eviction=True, decode_graph=False, engine_prefill_chunk_size=512,
        h2o_decode_budget=1024, h2o_decode_eviction_interval=128, h2o_prefill_budget=2048,
        max_model_len=8192,
        hf_config=SimpleNamespace(dtype=torch.bfloat16, max_position_embeddings=40960,
                                  head_dim=128),
    )
    _normalize_pace(config)
    assert config.pace_resolved_position_limit == 40960
    for name, value in overrides.items():
        setattr(config, name, value)
    with pytest.raises(error, match=message):
        _normalize_pace(config)


def test_bounded_pace_lifts_only_the_logical_length_limit():
    from sparseengine.method_registry import request_logical_length_limit

    config = SimpleNamespace(sparse_method="pace", pace_bounded_positions=True, max_model_len=8192)
    assert request_logical_length_limit(config) is None
    config.pace_bounded_positions = False
    assert request_logical_length_limit(config) == 8192
    assert request_logical_length_limit(
        SimpleNamespace(sparse_method="h2o", pace_bounded_positions=True, max_model_len=8192)
    ) == 8192


def test_rope_table_builder_matches_rotary_embedding_formula():
    from sparseengine.layers.rotary_embedding import RotaryEmbedding, build_rope_cos_sin_table

    for scaling in (None, (("rope_type", "yarn"), ("factor", 4.0),
                           ("original_max_position_embeddings", 256))):
        rotary = RotaryEmbedding(64, 64, 1024, 1e6, rope_scaling=scaling, backend="torch")
        table = build_rope_cos_sin_table(64, 1024, 1e6, scaling, device=torch.device("cpu"))
        assert torch.equal(table, rotary.cos_sin_cache.reshape(1024, -1))


def test_reference_release_ignores_columns_without_an_entry():
    store = PACEReferenceStore(capacity=4, num_kv_heads=1, head_dim=2,
                               dtype=torch.float32, device=torch.device("cpu"))
    store.in_use[torch.tensor([1, 3])] = True
    store.release(torch.tensor([-1, 3, -1], dtype=torch.int64))
    assert store.in_use.tolist() == [False, True, False, False]
    store.release(torch.full((5,), -1, dtype=torch.int64))
    assert store.in_use.tolist() == [False, True, False, False]


def test_pace_chain_snapshot_keeps_parent_metadata():
    from sparseengine.engine.cache_manager.chain_offload import ChainMethodState

    pace = _manager(PACECacheManager)
    _run(pace, PACEPlan(1, ((1, 3, (2,)),)), decode_steps=2)
    restored_parent = []
    with patch.object(H2OCacheManager, "snapshot_chain_method_state",
                      return_value=ChainMethodState(metadata={"h2o": 7})), \
         patch.object(H2OCacheManager, "restore_chain_method_state",
                      side_effect=lambda sid, st: restored_parent.append(st.metadata)):
        state = pace.snapshot_chain_method_state(0)
        before = {layer: pace.pace_row_state(layer, 0) for layer in (0, 1)}
        pace.restore_chain_method_state(0, state)
    assert restored_parent == [{"h2o": 7}]
    for layer in (0, 1):
        after = pace.pace_row_state(layer, 0)
        for name in ("logical", "bonus", "physical"):
            assert torch.equal(after[name], before[layer][name])


def test_pace_bounded_config_warns_about_compounding_rounding():
    from sparseengine.configs.sparse import _normalize_pace

    base = dict(
        sparse_method="pace", prefill_sparse_method="h2o_prefill",
        attention_cache_layout="explicit_kv", kv_cache_layout="token",
        enable_multimodal=False, pace_bonus_scale=SCALE, pace_bonus_decay=DECAY,
        pace_bounded_positions=True, pace_position_limit=None,
        h2o_decode_eviction=True, decode_graph=False, engine_prefill_chunk_size=512,
        h2o_decode_budget=1024, h2o_decode_eviction_interval=128, h2o_prefill_budget=2048,
        max_model_len=8192,
        hf_config=SimpleNamespace(dtype=torch.bfloat16, max_position_embeddings=40960,
                                  head_dim=128),
    )
    for tokens, expected in ((0, "compounds"), (16, "below the worst resident row"),
                             (4096, None)):
        with patch("sparseengine.configs.sparse.log_once") as warn:
            _normalize_pace(SimpleNamespace(**base, pace_reference_tokens=tokens))
        messages = [c.args[0] for c in warn.call_args_list]
        if expected is None:
            assert not messages
        else:
            assert any(expected in m for m in messages)


def test_full_reference_store_serves_planned_then_oldest_tokens_first():
    table = _cos_sin(64, 4)
    k_cache = torch.randn(6, 1, 4)
    store = PACEReferenceStore(capacity=2, num_kv_heads=1, head_dim=4,
                               dtype=torch.float32, device=torch.device("cpu"))
    pace = object.__new__(PACECacheManager)
    pace._pace_rows = {"bonus": {0: torch.tensor([0.0, 0.0, 6e4, 0.0, 9e4, 0.0])},
                       "logical": {0: torch.tensor([10, 3, 50, 7, 60, 1])}}
    entries, overflow = store.renumber(
        k_cache, torch.arange(6), torch.full((6,), -1, dtype=torch.int64),
        torch.arange(6) + 20, torch.arange(6), table,
        priority=pace._pace_reference_priority(0),
    )
    assert overflow == 4
    # planned columns 2 and 4 capture the two entries, not row-order columns 0 and 1
    assert (entries >= 0).nonzero().squeeze(1).tolist() == [2, 4]
    unordered, unordered_overflow = PACEReferenceStore(
        capacity=2, num_kv_heads=1, head_dim=4, dtype=torch.float32,
        device=torch.device("cpu")).renumber(
        k_cache.clone(), torch.arange(6), torch.full((6,), -1, dtype=torch.int64),
        torch.arange(6) + 20, torch.arange(6), table)
    # the priority is what changes the choice: without it, row order wins
    assert (unordered >= 0).nonzero().squeeze(1).tolist() == [0, 1]
    assert unordered_overflow == overflow
    pace._pace_rows["bonus"][0].zero_()
    entries, _ = PACEReferenceStore(capacity=2, num_kv_heads=1, head_dim=4,
                                    dtype=torch.float32, device=torch.device("cpu")).renumber(
        k_cache, torch.arange(6), torch.full((6,), -1, dtype=torch.int64),
        torch.arange(6), torch.arange(6), table,
        priority=pace._pace_reference_priority(0))
    # without a plan the oldest tokens (logical 1 and 3) capture first
    assert (entries >= 0).nonzero().squeeze(1).tolist() == [1, 5]


def test_rerotation_rejects_partial_rotary_tables():
    with pytest.raises(ValueError, match="full-width split-half RoPE"):
        rotate_keys(torch.zeros(2, 1, 8), _cos_sin(16, 4), torch.zeros(2), torch.ones(2))


def test_bounded_startup_rejects_a_rotary_table_that_does_not_match_the_keys():
    pace = _bounded_manager()
    pace._pace_validate_rotary()  # full-width table: accepted
    pace._pace_cos_sin = _cos_sin(128, pace.head_dim * 2)
    with pytest.raises(NotImplementedError, match="full-width split-half RoPE"):
        pace._pace_validate_rotary()


def _lookup(plan, positions, *, scale=SCALE, decay=DECAY):
    return compile_bonus(plan, scale=scale, decay=decay).lookup(
        torch.tensor(positions, dtype=torch.int64)).tolist()


def test_pace_bonus_weight_follows_the_decay_formula():
    for gap in (0, 1, 3, 10):
        weight = float(torch.tensor(0.5 + 0.5 * math.exp(-DECAY * gap), dtype=torch.float32))
        got, = _lookup(PACEPlan(5, ((0, 4, (5 + gap,)),)), [2])
        assert got == float(torch.tensor(weight, dtype=torch.float32) * SCALE)
    # a span needed at the current step gets exactly the full scale
    assert _lookup(PACEPlan(5, ((0, 4, (5,)),)), [0]) == [SCALE]


def test_pace_bonus_overlaps_take_the_maximum_and_adjacent_spans_do_not_overlap():
    near = float(torch.tensor(0.5 + 0.5 * math.exp(-DECAY * 1), dtype=torch.float32) * SCALE)
    far = float(torch.tensor(0.5 + 0.5 * math.exp(-DECAY * 8), dtype=torch.float32) * SCALE)
    overlap = PACEPlan(0, ((0, 10, (8,)), (4, 6, (1,))))
    assert _lookup(overlap, [3, 4, 5, 6]) == [far, near, near, far]
    adjacent = PACEPlan(0, ((0, 4, (8,)), (4, 8, (1,))))
    assert _lookup(adjacent, [3, 4, 7, 8]) == [far, near, near, 0.0]


def test_pace_bonus_ignores_passed_references_and_handles_an_empty_plan():
    assert _lookup(PACEPlan(9, ((0, 4, (2, 8)),)), [0, 3]) == [0.0, 0.0]
    empty = compile_bonus(PACEPlan(3), scale=SCALE, decay=DECAY)
    assert empty.empty and empty.breaks.numel() == 0
    assert empty.lookup(torch.tensor([0, 7])).tolist() == [0.0, 0.0]


def test_pace_bonus_lookup_at_the_boundaries():
    compiled = compile_bonus(PACEPlan(0, ((10, 20, (0,)),)), scale=SCALE, decay=DECAY)
    first, last = int(compiled.breaks[0]), int(compiled.breaks[-1])
    got = compiled.lookup(torch.tensor([first - 1, first, last - 1, last, last + 100])).tolist()
    assert got == [0.0, SCALE, SCALE, 0.0, 0.0]


def test_pace_plan_drops_spans_empty_after_clamping_and_ignores_span_order():
    assert PACEPlan(0, ((-5, 0, (1,)), (-5, -1, ()))).spans == ()
    a = PACEPlan(1, ((8, 9, (3,)), (0, 4, (2, 2)), (8, 9, (3,))))
    b = PACEPlan(1, ((0, 4, (2,)), (8, 9, (3,))))
    assert a == b and a.spans == ((0, 4, (2,)), (8, 9, (3,)))


def test_pace_plan_accepts_numpy_integers_but_not_bools():
    np = pytest.importorskip("numpy")
    plan = PACEPlan(np.int64(2), ((np.int32(0), np.int64(4), [np.int64(3)]),))
    assert plan == PACEPlan(2, ((0, 4, (3,)),))
    assert all(type(v) is int for v in (plan.current_step, *plan.spans[0][:2]))
    with pytest.raises(TypeError):
        PACEPlan(np.bool_(True))
    with pytest.raises(TypeError):
        PACEPlan(0, ((0, 4.0, (1,)),))


@pytest.mark.parametrize(("scale", "decay"), [(0.0, 0.35), (-1.0, 0.35), (float("nan"), 0.35),
                                              (1e5, -0.1), (1e5, float("inf"))])
def test_pace_bonus_rejects_scales_and_decays_that_break_the_ordering(scale, decay):
    with pytest.raises(ValueError):
        compile_bonus(PACEPlan(0, ((0, 4, (1,)),)), scale=scale, decay=decay)


@pytest.mark.parametrize("hf", [SimpleNamespace(partial_rotary_factor=0.5),
                                SimpleNamespace(rope_parameters={"partial_rotary_factor": 0.25}),
                                SimpleNamespace(rotary_dim=1)])
def test_bounded_startup_detects_partial_rotary_from_the_model_config(hf):
    pace = _bounded_manager()
    hf.head_dim, hf.model_type = pace.head_dim, "qwen3"
    pace.config.hf_config = hf
    with pytest.raises(NotImplementedError, match="declares rotary dimension"):
        pace._pace_validate_rotary()


def test_bounded_config_detects_partial_rotary():
    from sparseengine.configs.sparse import _normalize_pace

    config = SimpleNamespace(
        sparse_method="pace", prefill_sparse_method="h2o_prefill",
        attention_cache_layout="explicit_kv", kv_cache_layout="token",
        enable_multimodal=False, pace_bonus_scale=SCALE, pace_bonus_decay=DECAY,
        pace_bounded_positions=True, pace_position_limit=None, pace_reference_tokens=4096,
        h2o_decode_eviction=True, decode_graph=False, engine_prefill_chunk_size=512,
        h2o_decode_budget=1024, h2o_decode_eviction_interval=128, h2o_prefill_budget=2048,
        max_model_len=8192,
        hf_config=SimpleNamespace(dtype=torch.bfloat16, max_position_embeddings=40960,
                                  head_dim=128, partial_rotary_factor=0.5),
    )
    with pytest.raises(NotImplementedError, match="rotary dimension 64"):
        _normalize_pace(config)


def test_renumber_without_a_reference_store_warns():
    pace = _bounded_manager(reference_tokens=0)
    with patch("sparseengine.engine.cache_manager.methods.pace.log_once") as warn:
        _run(pace, None, prompt=8, chunk=4, decode_steps=0)
        key = (0, 0)
        pace._pace_renumber_row(0, 0, key, pace._pace_recorded(key), pace._pace_cos_sin)
    assert any("without a reference store" in c.args[0] for c in warn.call_args_list)


def test_pace_chain_snapshot_rejects_another_format_version():
    pace = _manager(PACECacheManager)
    _run(pace, None, decode_steps=1)
    state = pace.snapshot_chain_method_state(0)
    state.metadata["pace"]["version"] = 0
    with pytest.raises(RuntimeError, match="format 0"):
        pace.restore_chain_method_state(0, state)


@pytest.mark.parametrize(("hf", "expected"), [
    (SimpleNamespace(), 128),
    (SimpleNamespace(partial_rotary_factor=1.0), 128),
    (SimpleNamespace(partial_rotary_factor=0.5), 64),
    (SimpleNamespace(rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.25}), 32),
    (SimpleNamespace(rope_scaling={"rope_type": "yarn", "factor": 4.0}), 128),
    (SimpleNamespace(rotary_dim=64), 64),
    (SimpleNamespace(qk_rope_head_dim=64, partial_rotary_factor=1.0), 64),
])
def test_declared_rotary_dimension(hf, expected):
    from sparseengine.models.rope import resolve_declared_rotary_dim

    assert resolve_declared_rotary_dim(hf, 128) == expected


def test_bounded_startup_rejects_models_outside_the_split_half_families():
    pace = _bounded_manager()
    pace.config.hf_config.model_type = "glm4_moe_lite"
    with pytest.raises(NotImplementedError, match="split-half RoPE"):
        pace._pace_validate_rotary()


def test_pace_rejects_non_float32_scores():
    pace = _manager(PACECacheManager)
    seq = Sequence(list(range(8)))
    seq.seq_id, seq.current_chunk_size = 0, 4
    with pytest.raises(RuntimeError, match="float32"):
        pace._on_prefill_score_appended(0, seq, torch.zeros(4, dtype=torch.bfloat16),
                                        appended_from=0)


def test_mistral_runs_through_the_llama_relabel():
    """Mistral v0.3 has no SparseEngine model type of its own; a config relabelled to
    llama (Mistral-7B-Instruct-v0.3 shape) passes PACE's bounded-position checks."""
    from sparseengine.configs.sparse import _normalize_pace
    from sparseengine.models.spec import resolve_model_spec

    with pytest.raises(Exception, match="Unsupported SparseEngine model_type"):
        resolve_model_spec("mistral")
    mistral_as_llama = SimpleNamespace(
        model_type="llama", dtype=torch.bfloat16, head_dim=128, hidden_size=4096,
        num_attention_heads=32, num_key_value_heads=8, max_position_embeddings=32768,
        rope_theta=1e6, rope_scaling=None, sliding_window=None,
    )
    config = SimpleNamespace(
        sparse_method="pace", prefill_sparse_method="h2o_prefill",
        attention_cache_layout="explicit_kv", kv_cache_layout="token",
        enable_multimodal=False, pace_bonus_scale=SCALE, pace_bonus_decay=DECAY,
        pace_bounded_positions=True, pace_position_limit=8192, pace_reference_tokens=4096,
        h2o_decode_eviction=True, decode_graph=False, engine_prefill_chunk_size=1024,
        h2o_decode_budget=2048, h2o_decode_eviction_interval=128, h2o_prefill_budget=4096,
        max_model_len=32768, hf_config=mistral_as_llama,
    )
    _normalize_pace(config)
    assert config.pace_resolved_position_limit == 8192
    pace = _bounded_manager()
    pace.config.hf_config = SimpleNamespace(model_type="llama", head_dim=pace.head_dim,
                                            dtype=torch.float32)
    pace._pace_validate_rotary()
