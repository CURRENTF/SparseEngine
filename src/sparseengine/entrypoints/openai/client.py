import argparse
import json
import sys
import urllib.request
from collections.abc import Iterable
from typing import Any


def iter_sse_data(lines: Iterable[bytes | str]):
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data: "):
            raise ValueError(f"Unexpected SSE line: {line!r}")
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def chunk_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def print_stream_text(lines: Iterable[bytes | str]) -> str:
    output = ""
    for chunk in iter_sse_data(lines):
        text = chunk_text(chunk)
        if text:
            output += text
            print(text, end="", flush=True)
    print()
    return output


def _post_json(url: str, payload: dict[str, Any]):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request)


def main():
    parser = argparse.ArgumentParser(description="Call a Sparse-Engine OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chat", action="store_true", help="Use /v1/chat/completions.")
    parser.add_argument("--no-stream", action="store_true", help="Print the final JSON response.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if args.chat:
        url = f"{base_url}/chat/completions"
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": not args.no_stream,
        }
    else:
        url = f"{base_url}/completions"
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": not args.no_stream,
        }

    with _post_json(url, payload) as response:
        if args.no_stream:
            sys.stdout.write(response.read().decode("utf-8"))
            sys.stdout.write("\n")
            return
        print_stream_text(response)


if __name__ == "__main__":
    main()
