"""Protect portable experiment serialization and reject ambiguous output identities."""
import copy
import json

import pytest

from scripts.official_experiments.sparseengine_vs_vortex.prepare_runs import (
    build_outputs,
)


def build(cases, root, **variables):
    return build_outputs({'cases': cases}, {'OUTPUT_ROOT': str(root), **variables},
                         {'qwen'}, {'qwen': 0}, {'qwen': (1234, 1235)})


def native_case():
    return {'id': 'example', 'model_key': 'qwen', 'directory': 'qwen/example',
            'hyper_params': {'budget': 12},
            'job': {'command': ['python', '--config', '${HYPER_PARAMS_FILE}']}}


def test_nested_vortex_json_keeps_paths_as_single_arguments(tmp_path):
    # Paths containing quotes/spaces must survive both JSON serialization layers.
    case = {'id': 'example', 'model_key': 'qwen', 'directory': 'qwen/example',
            'vortex_config': {'module_path': '${MODEL_PATH}', 'budget': 12},
            'server_config': {'argv': ['--config', '${VORTEX_CONFIG_JSON}']},
            'job': {'command': ['python', '${SERVER_CONFIG_FILE}']}}
    model_path = str(tmp_path / 'a "quoted" model' / 'weights')
    result = build([case], tmp_path, MODEL_PATH=model_path)
    roundtrip = json.loads(json.dumps(result))
    argv = roundtrip['configs/example.server.json']['argv']
    assert len(argv) == 2
    assert json.loads(argv[1]) == {'module_path': model_path, 'budget': 12}
    assert roundtrip['qwen/jobs/example.json']['command'][1] == str(
        tmp_path / 'configs/example.server.json')


def test_missing_template_variable_cannot_create_a_partial_plan(tmp_path):
    case = native_case()
    case['job']['command'].append('${MISSING_MODEL}')
    with pytest.raises(KeyError, match='MISSING_MODEL'):
        build([case], tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('field,value', [('directory', '../escape'),
    ('directory', '/absolute/escape'), ('directory', 'glm/wrong_model'),
    ('id', '../escape')])
def test_output_identity_cannot_escape_its_case_directory(tmp_path, field, value):
    case = native_case()
    case[field] = value
    with pytest.raises(ValueError, match='Invalid case'):
        build([case], tmp_path)


@pytest.mark.parametrize('same_id', [True, False])
def test_duplicate_configs_or_job_targets_are_not_silently_overwritten(tmp_path, same_id):
    first = native_case()
    second = copy.deepcopy(first)
    if same_id:
        second['directory'] = 'qwen/different_job'
    else:
        second['id'] = 'different_config'
    with pytest.raises(ValueError, match='Duplicate case'):
        build([first, second], tmp_path)


def test_quality_smoke_failure_never_starts_full_cohort(tmp_path, monkeypatch):
    # Consolidating the phase runner must not turn a failed smoke into a full run.
    from types import SimpleNamespace
    from scripts.official_experiments.sparseengine_vs_vortex.session import run_quality
    prepared = tmp_path / 'prepared.json'
    prepared.write_text(json.dumps({'identity': {'token_buckets': [{'samples': 1}]},
        'samples': [{'sample': {'_id': 'a'}, 'prompt_tokens': 4, 'index': 0}]}))
    output = tmp_path / 'output'
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'output': str(output), 'prepared': str(prepared),
                               'cwd': str(tmp_path), 'command': ['model-runner']}))
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run_quality.subprocess, 'run', failed)
    with pytest.raises(RuntimeError, match='smoke failed'):
        run_quality.main(spec)
    assert len(calls) == 1
    assert not (output / 'full').exists()
    assert (output / 'status.tsv').read_text().startswith('smoke\t1\t')


def test_external_quality_failure_cleans_up_only_owned_server(tmp_path, monkeypatch):
    # The separate server process group must not outlive a failing quality job.
    from contextlib import nullcontext
    from types import SimpleNamespace
    from scripts.official_experiments.sparseengine_vs_vortex.session import serve_quality
    spec = tmp_path / 'server.json'
    spec.write_text(json.dumps({'output': str(tmp_path / 'output'),
        'server_command': ['server'], 'server_cwd': str(tmp_path),
        'server_url': 'http://127.0.0.1:1234', 'quality_spec': 'quality.json'}))
    sent, waited = [], []
    def popen(command, **kwargs):
        assert command == ['server'] and kwargs['start_new_session']
        return SimpleNamespace(pid=12345, poll=lambda: None,
                               wait=lambda timeout: waited.append(timeout))
    monkeypatch.setattr(serve_quality.subprocess, 'Popen', popen)
    monkeypatch.setattr(serve_quality.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(serve_quality, 'urlopen', lambda *a, **kw: nullcontext(SimpleNamespace(status=200)))
    monkeypatch.setattr(serve_quality.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(serve_quality.os, 'killpg', lambda pid, sig: sent.append((pid, sig)))
    with pytest.raises(RuntimeError, match='Quality exited'):
        serve_quality.main(spec)
    assert sent == [(12345, serve_quality.signal.SIGTERM)]
    assert waited == [30]
