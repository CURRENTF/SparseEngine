import json

import pytest

from benchmark.long_bench_v2.contracts import (
    aggregate_results,
    extract_answer,
    load_dataset,
    parse_token_buckets,
    render_prompt,
    select_samples,
)
from benchmark.sparseengine_regression.grading import grade_longbench_v2_quality
from benchmark.sparseengine_regression.run_suite import (
    _validated_longbench_v2_metrics,
)


def _sample(sample_id: str, *, answer: str = "A") -> dict[str, str]:
    return {
        "_id": sample_id,
        "domain": "Literature",
        "sub_domain": "Fiction",
        "difficulty": "easy",
        "length": "medium",
        "question": "Which choice is correct?",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": answer,
        "context": f"context for {sample_id}",
    }


def test_load_dataset_rejects_duplicate_sample_identity(tmp_path):
    path = tmp_path / "data.jsonl"
    row = _sample("duplicate")
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate _id"):
        load_dataset(path)


def test_official_prompt_and_answer_contract():
    template = "$DOC$\n$Q$\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$"
    prompt = render_prompt(template, _sample("one"))

    assert "context for one" in prompt
    assert "$DOC$" not in prompt
    assert extract_answer("The correct answer is (C)") == "C"
    assert extract_answer("The correct answer is D") == "D"
    assert extract_answer("C") is None


def test_prompt_rendering_preserves_dollar_tokens_from_official_content():
    template = "$DOC$\n$Q$\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$"
    sample = _sample("dollar-tokens")
    sample["context"] = "Use $W$ and keep the literal $Q$ variable unchanged."

    prompt = render_prompt(template, sample)

    assert "Use $W$ and keep the literal $Q$ variable unchanged." in prompt
    assert "Which choice is correct?" in prompt


def test_token_stratified_selection_is_deterministic_and_untruncated():
    rows = [_sample(f"sample-{index}") for index in range(8)]
    buckets = parse_token_buckets(
        [
            {
                "name": "small",
                "min_prompt_tokens": 10,
                "max_prompt_tokens": 19,
                "samples": 2,
            },
            {
                "name": "large",
                "min_prompt_tokens": 20,
                "max_prompt_tokens": 29,
                "samples": 2,
            },
        ]
    )

    def prepare(row):
        size = 12 if int(row["_id"].split("-")[-1]) < 4 else 24
        return row["context"], list(range(size))

    first = select_samples(
        rows,
        buckets=buckets,
        seed=7,
        prepare_prompt=prepare,
        max_prompt_tokens=29,
    )
    second = select_samples(
        list(reversed(rows)),
        buckets=buckets,
        seed=7,
        prepare_prompt=prepare,
        max_prompt_tokens=29,
    )

    assert [item["sample"]["_id"] for item in first] == [
        item["sample"]["_id"] for item in second
    ]
    assert [len(item["prompt_token_ids"]) for item in first] == [12, 12, 24, 24]
    assert [item["token_bucket"] for item in first] == [
        "small",
        "small",
        "large",
        "large",
    ]


def test_token_stratified_selection_fails_when_bucket_is_underfilled():
    buckets = parse_token_buckets(
        [
            {
                "name": "long",
                "min_prompt_tokens": 20,
                "max_prompt_tokens": 30,
                "samples": 2,
            }
        ]
    )

    with pytest.raises(ValueError, match="insufficient untruncated samples"):
        select_samples(
            [_sample("only")],
            buckets=buckets,
            seed=1,
            prepare_prompt=lambda row: (row["context"], list(range(25))),
            max_prompt_tokens=30,
        )


def test_aggregate_preserves_official_and_token_length_strata():
    rows = [
        {
            "status": "success",
            "correct": True,
            "difficulty": "easy",
            "official_length": "medium",
            "token_bucket": "32k-64k",
            "domain": "Literature",
        },
        {
            "status": "success",
            "correct": False,
            "difficulty": "hard",
            "official_length": "long",
            "token_bucket": "64k-96k",
            "domain": "History",
        },
    ]

    aggregate = aggregate_results(rows)

    assert aggregate["accuracy"] == 50.0
    assert aggregate["by_official_length"]["long"]["samples"] == 1
    assert aggregate["by_token_bucket"]["32k-64k"]["accuracy"] == 100.0


def test_aggregate_counts_unparseable_model_output_as_official_incorrect_answer():
    rows = [
        {
            "status": "success",
            "correct": True,
            "difficulty": "easy",
            "official_length": "short",
            "token_bucket": "32k-64k",
            "domain": "Single-Document QA",
        },
        {
            "status": "parse_failed",
            "predicted_answer": None,
            "correct": False,
            "difficulty": "hard",
            "official_length": "medium",
            "token_bucket": "64k-96k",
            "domain": "Long Structured Data Understanding",
        },
    ]

    aggregate = aggregate_results(rows)

    assert aggregate["status"] == "success"
    assert aggregate["evaluated_samples"] == 2
    assert aggregate["successful_samples"] == 1
    assert aggregate["parse_failed_samples"] == 1
    assert aggregate["failed_samples"] == 0
    assert aggregate["accuracy"] == 50.0


def test_regression_gate_accepts_explicit_parse_failure_as_incorrect(tmp_path):
    rows = [
        {
            "index": 0,
            "status": "success",
            "predicted_answer": "A",
            "correct": True,
            "token_bucket": "32k-64k",
            "prompt_tokens": 40_000,
        },
        {
            "index": 1,
            "status": "parse_failed",
            "predicted_answer": None,
            "correct": False,
            "token_bucket": "64k-96k",
            "prompt_tokens": 80_000,
        },
    ]
    (tmp_path / "sample_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "status": "success",
                "samples": 2,
                "evaluated_samples": 2,
                "failed_samples": 0,
                "accuracy": 50.0,
            }
        ),
        encoding="utf-8",
    )

    aggregate = _validated_longbench_v2_metrics(
        tmp_path,
        token_buckets=[
            {
                "name": "32k-64k",
                "min_prompt_tokens": 32_768,
                "max_prompt_tokens": 65_535,
                "samples": 1,
            },
            {
                "name": "64k-96k",
                "min_prompt_tokens": 65_536,
                "max_prompt_tokens": 98_303,
                "samples": 1,
            },
        ],
    )

    assert aggregate["accuracy"] == 50.0


def test_longbench_v2_grade_requires_baseline_and_bounds_sparse_loss():
    assert grade_longbench_v2_quality(
        50.0,
        42.0,
        minimum_vanilla_score=25.0,
        maximum_score_loss=10.0,
    ).grade == "C"
    assert grade_longbench_v2_quality(
        20.0,
        20.0,
        minimum_vanilla_score=25.0,
        maximum_score_loss=10.0,
    ).grade == "D"
    assert grade_longbench_v2_quality(
        50.0,
        30.0,
        minimum_vanilla_score=25.0,
        maximum_score_loss=10.0,
    ).grade == "D"
