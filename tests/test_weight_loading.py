import threading

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sparseengine.utils import loader
from sparseengine.utils.weight_target import WeightTarget


class _TwoShardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(2, 2, bias=False)
        self.right = nn.Linear(2, 2, bias=False)


class _BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2))
        self.register_buffer("clip_max", torch.tensor(float("inf")))


class _RankLocalWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))
        self.weight.weight_loader = self.weight_loader

    def rank_local_weight_slice(self, source_shape, **_):
        assert tuple(source_shape) == (4, 2)
        return (slice(2, 4), slice(None))

    @staticmethod
    def weight_loader(param, loaded_weight):
        assert tuple(loaded_weight.shape) == (2, 2)
        param.data.copy_(loaded_weight)


class _RankLocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _RankLocalWeight()


class _SpecialRankLocalModel(nn.Module):
    special_weight_loaders = (".expert_weight",)

    def __init__(self):
        super().__init__()
        self.experts = _RankLocalWeight()

    @staticmethod
    def map_weight_name(source_weight_name):
        if source_weight_name == "expert.weight":
            return "experts.gate.expert_weight"
        return source_weight_name

    def resolve_special_weight(self, target_weight_name):
        if target_weight_name != "experts.gate.expert_weight":
            return None
        return WeightTarget(self.experts, (0, "gate"))

    def load_special_weight(self, target_weight_name, loaded_weight, loaded_scale):
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        assert loaded_scale is None
        target.module.weight_loader(target.module.weight, loaded_weight)
        return 1


class _ExpertOwnershipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Linear(2, 2, bias=False)
        self.skipped = []

    @staticmethod
    def map_weight_name(source_weight_name):
        if source_weight_name.startswith("remote."):
            return None
        return source_weight_name

    def record_skipped_weight(
        self,
        source_weight_name,
        loaded_weight_shape,
        loaded_weight_dtype,
        loaded_scale_shape,
        loaded_scale_dtype,
    ):
        self.skipped.append(
            (
                source_weight_name,
                loaded_weight_shape,
                loaded_weight_dtype,
                loaded_scale_shape,
                loaded_scale_dtype,
            )
        )


def _write_two_shards(path):
    left = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    right = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
    save_file({"left.weight": left}, path / "model-00001-of-00002.safetensors")
    save_file({"right.weight": right}, path / "model-00002-of-00002.safetensors")
    return left, right


def test_load_model_prefetches_shards_in_parallel(tmp_path, monkeypatch, capsys):
    expected_left, expected_right = _write_two_shards(tmp_path)
    model = _TwoShardModel()
    original_read = loader._read_safetensors_shard
    both_workers_started = threading.Barrier(2, timeout=5)
    worker_names = set()
    worker_names_lock = threading.Lock()

    def synchronized_read(path, model=None):
        with worker_names_lock:
            worker_names.add(threading.current_thread().name)
        both_workers_started.wait()
        return original_read(path, model)

    monkeypatch.setattr(loader, "_read_safetensors_shard", synchronized_read)

    loader.load_model(model, str(tmp_path), num_threads=2)

    assert len(worker_names) == 2
    torch.testing.assert_close(model.left.weight, expected_left)
    torch.testing.assert_close(model.right.weight, expected_right)
    assert "Multi-thread loading shards" in capsys.readouterr().err


def test_load_model_rejects_non_positive_thread_count(tmp_path):
    _write_two_shards(tmp_path)
    with pytest.raises(ValueError, match="num_threads must be positive"):
        loader.load_model(_TwoShardModel(), str(tmp_path), num_threads=0)


def test_load_model_restores_checkpoint_buffers(tmp_path):
    save_file(
        {"weight": torch.ones(2), "clip_max": torch.tensor(3.5)},
        tmp_path / "model.safetensors",
    )
    model = _BufferModel()

    loader.load_model(model, str(tmp_path), show_progress=False)

    torch.testing.assert_close(model.clip_max, torch.tensor(3.5))


def test_load_model_reads_only_rank_local_tensor_slice(tmp_path):
    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    save_file({"proj.weight": full_weight}, tmp_path / "model.safetensors")
    model = _RankLocalModel()

    loader.load_model(model, str(tmp_path), tp_rank=1, tp_size=2)

    torch.testing.assert_close(model.proj.weight, full_weight[2:4])


def test_special_weight_resolves_rank_local_slice_from_target_module(tmp_path):
    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    save_file({"expert.weight": full_weight}, tmp_path / "model.safetensors")
    model = _SpecialRankLocalModel()

    loader.load_model(model, str(tmp_path), tp_rank=1, tp_size=2)

    torch.testing.assert_close(model.experts.weight, full_weight[2:4])


def test_load_model_keeps_remote_expert_tensors_as_metadata(tmp_path):
    local_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file(
        {
            "local.weight": local_weight,
            "remote.weight": torch.ones(4, 2),
            "remote.weight_scale_inv": torch.ones(1, 1),
        },
        tmp_path / "model.safetensors",
    )
    model = _ExpertOwnershipModel()

    shard = loader._read_safetensors_shard(
        str(tmp_path / "model.safetensors"),
        model,
    )
    assert set(shard.metadata) == {
        "local.weight",
        "remote.weight",
        "remote.weight_scale_inv",
    }
    assert set(shard.tensors) == {"local.weight"}

    loader.load_model(model, str(tmp_path), num_threads=2)

    torch.testing.assert_close(model.local.weight, local_weight)
    assert model.skipped == [
        ("remote.weight", (4, 2), "F32", (1, 1), "F32")
    ]


def test_load_model_selects_all_files_for_local_checkpoint_rank(tmp_path):
    rank0_left = torch.full((2, 2), 10.0)
    rank0_right = torch.full((2, 2), 11.0)
    rank1_left = torch.full((2, 2), 20.0)
    rank1_right = torch.full((2, 2), 21.0)
    save_file(
        {"left.weight": rank0_left},
        tmp_path / "model0-mp2-00001.safetensors",
    )
    save_file(
        {"right.weight": rank0_right},
        tmp_path / "model0-mp2-00002.safetensors",
    )
    save_file(
        {"left.weight": rank1_left},
        tmp_path / "model1-mp2-00001.safetensors",
    )
    save_file(
        {"right.weight": rank1_right},
        tmp_path / "model1-mp2-00002.safetensors",
    )
    model = _TwoShardModel()

    loader.load_model(model, str(tmp_path), tp_rank=1, tp_size=2, num_threads=2)

    torch.testing.assert_close(model.left.weight, rank1_left)
    torch.testing.assert_close(model.right.weight, rank1_right)


class _AliasedBlockScaleModel(nn.Module):
    checkpoint_block_scale_suffixes = (".scale",)

    def __init__(self):
        super().__init__()
        self.proj = _AliasedBlockScaleWeight()


class _AliasedBlockScaleWeight(nn.Module):
    quantized = True
    _quantized_weight_loaded = False

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))
        self.register_buffer("scale", torch.empty(2, 2))

    def rank_local_weight_slice(self, source_shape, *, is_scale, **_):
        assert tuple(source_shape) == (4, 2)
        return (slice(2, 4), slice(None))

    def load_quantized_weight(self, weight, scale, shard_id):
        assert shard_id is None
        self.weight.data.copy_(weight)
        self.scale.copy_(scale)
        self._quantized_weight_loaded = True


def test_declared_block_scale_alias_loads_rank_local_weight_and_scale(tmp_path):
    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    scale = weight + 20
    save_file({"proj.weight": weight, "proj.scale": scale}, tmp_path / "model.safetensors")
    model = _AliasedBlockScaleModel()
    loader.load_model(model, str(tmp_path), num_threads=1)
    torch.testing.assert_close(model.proj.weight, weight[2:])
    torch.testing.assert_close(model.proj.scale, scale[2:])


def test_declared_block_scale_alias_rejects_ambiguous_scale(tmp_path):
    save_file({"proj.weight": torch.ones(4, 2), "proj.scale": torch.ones(4, 2),
               "proj.weight_scale_inv": torch.ones(4, 2)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="duplicate"):
        loader.load_model(_AliasedBlockScaleModel(), str(tmp_path), num_threads=1)
