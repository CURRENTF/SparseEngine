"""Regression: serialized FP8 checkpoints may omit optional generation metadata."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmark.long_bench.pred import load_model_and_tokenizer
from benchmark.long_bench_v2.pred import _eos_token_ids


@pytest.mark.parametrize('malformed', [False, True])
@pytest.mark.parametrize('version', [1, 2])
def test_longbench_missing_generation_metadata_preserves_eos_but_bad_file_fails(tmp_path, malformed, version):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'llama', 'eos_token_id': [2, 3]}))
    if malformed:
        (tmp_path / 'generation_config.json').write_text('{bad json')
    args = SimpleNamespace(model_path=str(tmp_path), tokenizer_path=str(tmp_path),
                           deltakv_checkpoint_path=None, sparse_method='vanilla', max_model_len=1024)
    tokenizer = SimpleNamespace(eos_token_id=3, eot_token_id=4)
    if version == 2:
        if malformed:
            with pytest.raises(OSError):
                _eos_token_ids(str(tmp_path), tokenizer)
        else:
            assert _eos_token_ids(str(tmp_path), tokenizer) == [2, 3, 4]
        return
    with patch('benchmark.long_bench.pred.get_sparseengine_generate_api'), patch(
        'benchmark.long_bench.pred.AutoTokenizer.from_pretrained', return_value=tokenizer
    ):
        if malformed:
            with pytest.raises(OSError):
                load_model_and_tokenizer(0, args, {})
        else:
            assert load_model_and_tokenizer(0, args, {})[3] == [2, 3, 4]


@pytest.mark.parametrize('generation_eos', [None, [5, 3]])
def test_longbench_v2_nested_eos_preserved_and_generation_config_takes_precedence(
    tmp_path, generation_eos
):
    # Composite checkpoints keep EOS in text_config, unlike the flat Llama fixture.
    (tmp_path / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3_5',
        'text_config': {'model_type': 'qwen3_5_text', 'eos_token_id': [2, 3]},
    }))
    if generation_eos is not None:
        (tmp_path / 'generation_config.json').write_text(
            json.dumps({'eos_token_id': generation_eos})
        )
    tokenizer = SimpleNamespace(eos_token_id=3, eot_token_id=4)

    expected = [2, 3, 4] if generation_eos is None else [5, 3, 4]
    assert _eos_token_ids(str(tmp_path), tokenizer) == expected
