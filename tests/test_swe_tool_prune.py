from copy import deepcopy

import pytest

from benchmark.swe_bench_lite.tool_prune import tool_result_ranges


class PairTokenizer:
    bos_token = None

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)],
        }

    def encode(self, text, **kwargs):
        return [text[i:i + 2] for i in range(0, len(text), 2)]


def render(chat):
    return "".join(
        f"[{m['role']}]" + m.get("reasoning_content", "")
        + str(m.get("tool_calls", "")) + m["content"].strip() + "[/message]"
        for m in chat["messages"]
    ) + "[assistant]"


def test_only_tool_bodies_are_selected_despite_repeated_text_and_boundary_tokens():
    chat = {"messages": [
        {"role": "user", "content": "重复 identical result"},
        {"role": "assistant", "content": "answer", "reasoning_content": "thinking", "tool_calls": "bash(command)"},
        {"role": "tool", "content": "  重复 identical result\n"},
        {"role": "tool", "content": "重复 identical result"},
    ]}
    original = deepcopy(chat)
    prompt = render(chat)
    token_count = len(PairTokenizer().encode(prompt))
    result = tool_result_ranges(chat, PairTokenizer(), render, block_size=1, usable_tokens=token_count)
    expected = []
    for i, token in enumerate(result["token_ids"]):
        start = i * 2
        # Independent oracle uses explicit rendered role wrappers, not annotation markers.
        is_tool = False
        cursor = 0
        while (left := prompt.find("[tool]", cursor)) >= 0:
            left += len("[tool]")
            right = prompt.index("[/message]", left)
            is_tool |= left <= start and start + len(token) <= right
            cursor = right + len("[/message]")
        if is_tool:
            expected.append(i)
    actual = [i for left, right in result["ranges"] for i in range(left, right)]
    assert actual == expected
    assert len(result["ranges"]) == 2
    assert chat == original


def test_block_alignment_and_cached_prefix_limit_keep_mixed_blocks():
    chat = {"messages": [{"role": "tool", "content": "x" * 90}]}
    result = tool_result_ranges(chat, PairTokenizer(), render, block_size=4, usable_tokens=24)
    assert result["ranges"] == [(4, 24)]
    assert result["eligible_tokens"] == 20


def test_marker_sensitive_template_fails_instead_of_guessing_boundaries():
    chat = {"messages": [{"role": "tool", "content": "result"}]}
    def unusual_render(value):
        return render(value) + str(len(value["messages"][0]["content"]))
    with pytest.raises(ValueError, match="changed the rendered prompt"):
        tool_result_ranges(chat, PairTokenizer(), unusual_render, block_size=1, usable_tokens=1)


def test_no_tool_results_produces_no_deletion_ranges():
    result = tool_result_ranges(
        {"messages": [{"role": "user", "content": "keep me"}]}, PairTokenizer(), render,
        block_size=1, usable_tokens=10,
    )
    assert result["ranges"] == []


def test_local_fast_tokenizer_uses_shared_server_rendering(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from benchmark.swe_bench_lite.tool_prune import ToolResultPruneSelector

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "result": 1, "keep": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m.role }} : {{ m.content | trim }} ; {% endfor %}assistant :"
    tokenizer.save_pretrained(tmp_path)
    selector = ToolResultPruneSelector(str(tmp_path))
    result = selector.select(
        {"model": "local-test", "messages": [{"role": "user", "content": "keep"},
                                               {"role": "tool", "tool_call_id": "call-1", "content": "result result"}]},
        block_size=1, usable_tokens=8,
    )
    assert result["ranges"] == [(6, 8)]
    assert result["token_ids"][6:8] == [1, 1]


def test_incremental_selector_never_selects_old_identical_tool_body():
    old = {'role': 'tool', 'content': 'identical body ' * 5}
    chat = {'messages': [old, {'role': 'assistant', 'content': 'thinking'}, dict(old)]}
    tokenizer = PairTokenizer()
    prompt = render(chat)
    result = tool_result_ranges(chat, tokenizer, render, block_size=1,
                                usable_tokens=len(tokenizer.encode(prompt)), message_start=2)
    left = prompt.rindex('[tool]') + len('[tool]')
    right = prompt.index('[/message]', left)
    actual = [i for l,r in result['ranges'] for i in range(l,r)]
    expected = [i for i,t in enumerate(result['token_ids']) if left <= i*2 and i*2+len(t) <= right]
    assert actual == expected
    assert chat['messages'][0] == old


def test_selected_tool_message_excludes_newer_tool_results():
    chat = {'messages': [
        {'role': 'tool', 'content': 'older result ' * 5},
        {'role': 'assistant', 'content': 'next call'},
        {'role': 'tool', 'content': 'newer result ' * 5},
    ]}
    tokenizer = PairTokenizer()
    prompt = render(chat)
    result = tool_result_ranges(
        chat, tokenizer, render, block_size=1,
        usable_tokens=len(tokenizer.encode(prompt)), message_indices=(0,),
    )
    old_start = prompt.index('[tool]') + len('[tool]')
    old_end = prompt.index('[/message]', old_start)
    selected = [i for left, right in result['ranges'] for i in range(left, right)]
    assert selected
    assert all(old_start <= i * 2 and (i + 1) * 2 <= old_end for i in selected)
