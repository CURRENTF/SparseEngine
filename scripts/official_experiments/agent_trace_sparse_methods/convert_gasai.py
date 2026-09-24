#!/usr/bin/env python3
"""Convert a frozen Gasai selection to the existing forced agent replay format."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmark.sparseengine_regression.agent_trace import (
    digest, load_trace, prepare_forced_agent, read, replay_body, write,
)
from benchmark.swe_bench_lite.agent_trace import SCHEMA
from sparseengine.engine.input_processor import tokenize_text_prompt
from sparseengine.entrypoints.openai.protocol.chat import ChatCompletionRequest, ChatMessage
from sparseengine.entrypoints.openai.render import (
    _chat_prompt, _chat_request_prompt, resolve_chat_template_kwargs, resolve_chat_tools,
)


ASSISTANT = "<|assistant|>"
EOS = "<|eos|>"
TOOL_RESULT = "<|tool_result|>"
SYSTEM = "<|bos|><|system|>"
USER = "<|user|>"
TEMPLATE_KWARGS = {"clear_thinking": False}


def parse_trajectory(serialized):
    if not serialized.startswith(SYSTEM) or not serialized.endswith(EOS):
        raise ValueError("Gasai trajectory must start with a system message and end with EOS")
    user_at = serialized.index(USER, len(SYSTEM))
    assistant_at = serialized.index(ASSISTANT, user_at + len(USER))
    system = serialized[len(SYSTEM):user_at]
    if "<|tools|>" not in system:
        raise ValueError("Gasai system message omits tool declarations")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": serialized[user_at + len(USER):assistant_at]}]
    turns = []
    tool_results = 0
    position = assistant_at
    while position < len(serialized):
        if not serialized.startswith(ASSISTANT, position):
            raise ValueError(f"Expected assistant marker at character {position}")
        end = serialized.index(EOS, position + len(ASSISTANT))
        answer = serialized[position + len(ASSISTANT):end]
        if not answer:
            raise ValueError("Empty Gasai assistant answer")
        turns.append((list(messages), answer))
        messages.append({"role": "assistant", "content": answer})
        position = end + len(EOS)
        if position == len(serialized):
            break
        if not serialized.startswith(TOOL_RESULT, position):
            raise ValueError(f"Expected tool-result marker at character {position}")
        end = serialized.index(EOS, position + len(TOOL_RESULT))
        # Preserve the source marker and payload as text. No tools are executed.
        messages.append({"role": "user", "content": serialized[position:end]})
        tool_results += 1
        position = end + len(EOS)
        if position == len(serialized):
            raise ValueError("Trajectory ends with a tool result instead of an assistant answer")
    if tool_results != len(turns) - 1:
        raise ValueError("Gasai assistant/tool-result alternation is incomplete")
    return turns


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def distribution(values):
    return {"min": min(values), "p10": percentile(values, .1),
            "p50": percentile(values, .5), "p90": percentile(values, .9),
            "max": max(values)}


def convert(source, selection_path, model, output, max_model_len, max_output_tokens):
    selection = read(selection_path)
    items = selection["items"]
    if (selection.get("selected_trajectories") != len(items) or not items
            or selection.get("assistant_requests") != sum(item["assistant_turns"] for item in items)
            or [item["draw_index"] for item in items] != list(range(len(items)))):
        raise ValueError("Selection manifest count or draw order is inconsistent")
    rows = [item["source_row_zero_based"] for item in items]
    ids = [item["trajectory_id"] for item in items]
    if len(set(rows)) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("Selection contains duplicate source rows or trajectory IDs")
    if digest(source) != selection["source_sha256"]:
        raise ValueError("Source JSONL hash differs from frozen selection")
    wanted = dict(zip(rows, items))
    records = {}
    with source.open(encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            if row in wanted:
                record = json.loads(line)
                if record["trajectory_id"] != wanted[row]["trajectory_id"]:
                    raise ValueError(f"Trajectory ID mismatch at source row {row}")
                records[row] = record
    if len(records) != len(items) or row + 1 != selection["source_rows"]:
        raise ValueError("Selected records or source row count missing")

    model = model.resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, local_files_only=True)
    vocabulary = int(read(model / "config.json")["vocab_size"])
    output.mkdir(parents=True, exist_ok=False)
    trace = output / "trace"
    trace.mkdir()
    agents = []
    prepared_agents = {}
    workload_digest = hashlib.sha256()
    full_lengths = []
    output_lengths = []
    expected_tokens = 0

    for number, item in enumerate(items):
        iid = item["trajectory_id"]
        serialized = records[item["source_row_zero_based"]]["gasai"]
        if len(serialized) != item["chars"]:
            raise ValueError(f"Source character count mismatch: {iid}")
        parsed = parse_trajectory(serialized)
        if len(parsed) != item["assistant_turns"] or len(parsed) - 1 != item["tool_results"]:
            raise ValueError(f"Source assistant/tool counts mismatch: {iid}")
        turns = []
        for turn, (messages, _) in enumerate(parsed):
            body = {"model": "forced-trace", "messages": messages, "stream": False,
                    "n": 1, "chat_template_kwargs": TEMPLATE_KWARGS}
            turns.append({"turn": turn, "request": body, "response": {"usage": {"completion_tokens": 1}},
                          "think_time_s": None if turn == 0 else 0.0, "completion_tokens": 1})
        agent = {"instance_id": iid, "exit_status": "source_final_answer", "turns": turns}
        prepared = prepare_forced_agent(agent, tokenizer, "forced-trace")

        last_request = ChatCompletionRequest.model_validate(replay_body(turns[-1], "forced-trace"))
        prompt = tokenize_text_prompt(tokenizer, _chat_request_prompt(tokenizer, last_request))
        final_messages = last_request.messages + [ChatMessage(role="assistant", content=parsed[-1][1])]
        full = tokenize_text_prompt(tokenizer, _chat_prompt(
            tokenizer, final_messages, resolve_chat_template_kwargs(last_request),
            resolve_chat_tools(last_request), add_generation_prompt=False))
        if full[:len(prompt)] != prompt or len(full) == len(prompt):
            raise ValueError(f"Final answer is not a token-prefix extension: {iid}")
        full_lengths.append(len(full))
        if len(full) > max_model_len:
            raise ValueError(f"Full trajectory exceeds max model length: {iid}")

        for turn, spec in zip(turns, prepared):
            token_ids = spec["token_ids"]
            count = len(token_ids) if token_ids is not None else len(full) - len(prompt)
            if count <= 0 or count > max_output_tokens:
                raise ValueError(f"Output count outside configured bound: {iid} turn {turn['turn']}: {count}")
            if token_ids is not None and any(token < 0 or token >= vocabulary for token in token_ids):
                raise ValueError(f"Forced token outside target vocabulary: {iid} turn {turn['turn']}")
            turn["completion_tokens"] = count
            turn["response"]["usage"]["completion_tokens"] = count
            output_lengths.append(count)
            expected_tokens += count
            workload_digest.update(json.dumps([iid, turn["turn"], token_ids, count],
                                              separators=(",", ":")).encode())
        filename = f"agent_{number:03d}.json"
        write(trace / filename, agent)
        agents.append({"instance_id": iid, "file": filename, "sha256": digest(trace / filename),
                       "turns": len(turns), "exit_status": agent["exit_status"],
                       "source_row_zero_based": item["source_row_zero_based"],
                       "rendered_full_tokens": len(full)})
        prepared_agents[filename] = prepared
        print(json.dumps({"converted": number + 1, "total": len(items), "turns": len(turns),
                          "full_tokens": len(full)}, ensure_ascii=False), flush=True)

    manifest = {"schema": SCHEMA, "timing_boundary": "none_source_has_no_request_timestamps",
                "timing_quality": "synthetic", "source_repo": selection["source_repo"],
                "source_revision": selection["source_revision"], "source_sha256": selection["source_sha256"],
                "selection_sha256": digest(selection_path),
                "selection": selection["sampling"], "selection_seed": selection["seed"],
                "source_format": "gasai_serialized_transcript",
                "tool_result_mapping": "user_message_with_original_tool_result_marker_no_tool_execution",
                "chat_template_kwargs": TEMPLATE_KWARGS,
                "agents": agents, "instance_count": len(agents),
                "request_count": sum(entry["turns"] for entry in agents)}
    write(trace / "manifest.json", manifest)
    load_trace(trace)
    write(output / "forced_glm.json", {
        "schema": "agent_forced_workload_v1", "trace_sha256": digest(trace / "manifest.json"),
        "model_path": str(model), "forced_workload_sha256": workload_digest.hexdigest(),
        "instance_count": len(agents), "request_count": manifest["request_count"],
        "expected_completion_tokens": expected_tokens, "agents": prepared_agents})
    summary = {"source_sha256": selection["source_sha256"],
               "selection_sha256": digest(selection_path),
               "trace_manifest_sha256": digest(trace / "manifest.json"),
               "forced_workload_sha256": digest(output / "forced_glm.json"),
               "model_path": str(model), "instance_count": len(agents),
               "request_count": manifest["request_count"],
               "expected_completion_tokens": expected_tokens,
               "max_model_len": max_model_len, "max_output_tokens": max_output_tokens,
               "assistant_turns_per_trajectory": distribution([entry["turns"] for entry in agents]),
               "rendered_full_tokens_per_trajectory": distribution(full_lengths),
               "output_tokens_per_turn": distribution(output_lengths)}
    write(output / "conversion_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=202752)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    args = parser.parse_args()
    if args.max_model_len <= 0 or args.max_output_tokens <= 0:
        parser.error("Model and output limits must be positive")
    print(json.dumps(convert(args.source_jsonl, args.selection_manifest, args.model, args.output,
                             args.max_model_len, args.max_output_tokens), ensure_ascii=False))


if __name__ == "__main__":
    main()
