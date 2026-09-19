import os
import json
import argparse
import numpy as np
import traceback
from collections import Counter
from pathlib import Path

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = os.getenv("SPARSEENGINE_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

TASK_HIERARCHY = {
    "SDQA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "MDQA": ["hotpotqa", "2wikimqa", "musique"],
    "SUM": ["gov_report", "qmsum", "multi_news"],
    "FewShot": ["trec", "triviaqa", "samsum"],
    "Syn": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--deltakv_checkpoint_path", type=str, default=None)
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--path", type=str, default=None, help="The path to the prediction results")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def _round_float(value):
    return round(float(value), 2)


def aggregate_category_scores(task_scores):
    category_scores = {}
    for category, tasks in TASK_HIERARCHY.items():
        present_tasks = [task for task in tasks if task in task_scores]
        if not present_tasks:
            continue

        first_score = task_scores[present_tasks[0]]
        if isinstance(first_score, dict):
            bucket_scores = {}
            for bucket in first_score:
                values = [
                    task_scores[task][bucket]
                    for task in present_tasks
                    if isinstance(task_scores[task], dict) and bucket in task_scores[task]
                ]
                if values:
                    bucket_scores[bucket] = _round_float(np.mean(values))
            category_scores[category] = bucket_scores
        else:
            values = [
                task_scores[task]
                for task in present_tasks
                if not isinstance(task_scores[task], dict)
            ]
            if values:
                category_scores[category] = _round_float(np.mean(values))

    if not category_scores:
        return category_scores, None

    first_category_score = next(iter(category_scores.values()))
    if isinstance(first_category_score, dict):
        overall_score = {}
        for bucket in first_category_score:
            values = [
                score[bucket]
                for score in category_scores.values()
                if isinstance(score, dict) and bucket in score
            ]
            if values:
                overall_score[bucket] = _round_float(np.mean(values))
    else:
        values = [score for score in category_scores.values() if not isinstance(score, dict)]
        overall_score = _round_float(np.mean(values)) if values else None

    return category_scores, overall_score


if __name__ == '__main__':
    args = parse_args()
    
    if args.path:
        path = args.path
    else:
        deltakv_checkpoint_path = args.deltakv_checkpoint_path if args.deltakv_checkpoint_path is not None else args.cfg
        if deltakv_checkpoint_path is not None:
            compressor_name = os.path.basename(deltakv_checkpoint_path.rstrip('/'))
        else:
            compressor_name = "None"
            
        if args.e:
            path = os.path.join(BASE_PATH, f"benchmark/long_bench/pred_e/{args.model}/{compressor_name}")
        else:
            path = os.path.join(BASE_PATH, f"benchmark/long_bench/pred/{args.model}/{compressor_name}")
    
    task_scores = dict()
    task_statuses = dict()
    failed_tasks = []
    if not os.path.exists(path):
        
        print(f"Path {path} does not exist.")
        exit(1)
        
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        if dataset not in dataset2metric:
            continue
        status_counts: Counter[str] = Counter()
        invalid_statuses: Counter[str] = Counter()
        metric_failure_records = []
        all_classes = None
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)
                status = data.get("status")
                if status is None:
                    invalid_statuses["missing"] += 1
                    status = "parse_failed"
                if status not in SAMPLE_STATUSES:
                    invalid_statuses[status] += 1
                    status = "parse_failed"
                status_counts[status] += 1
                if status != "success":
                    continue
                try:
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
                except Exception as exc:
                    status_counts["metric_failed"] += 1
                    metric_failure_records.append(
                        {
                            "dataset": dataset,
                            "line_idx": line_idx,
                            "status": "metric_failed",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

        non_success = sum(
            count
            for status, count in status_counts.items()
            if status not in {"success", "skipped_by_policy"}
        )
        if invalid_statuses:
            failed_tasks.append(dataset)
            task_statuses[dataset] = {
                "status": "metric_failed",
                "status_counts": dict(status_counts),
                "invalid_statuses": dict(invalid_statuses),
                "error": f"Invalid sample statuses encountered: {dict(invalid_statuses)}",
            }
            continue
        if metric_failure_records:
            failed_tasks.append(dataset)
            task_statuses[dataset] = {
                "status": "metric_failed",
                "status_counts": dict(status_counts),
                "metric_failure_records": metric_failure_records,
            }
            continue
        if not predictions:
            only_skipped = status_counts and set(status_counts) <= {"skipped_by_policy"}
            task_statuses[dataset] = {
                "status": "skipped_by_policy" if only_skipped else "metric_failed",
                "status_counts": dict(status_counts),
                "error": None if only_skipped else "No successful predictions to score.",
            }
            if not only_skipped:
                failed_tasks.append(dataset)
            continue

        if args.e and len(lengths) != len(predictions):
            failed_tasks.append(dataset)
            task_statuses[dataset] = {
                "status": "metric_failed",
                "status_counts": dict(status_counts),
                "error": (
                    f"LongBench-E requires one length per prediction, got "
                    f"{len(lengths)} lengths for {len(predictions)} predictions."
                ),
            }
            continue

        try:
            if args.e:
                score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            else:
                score = scorer(dataset, predictions, answers, all_classes)
        except Exception as exc:
            failed_tasks.append(dataset)
            metric_failed = []
            for idx in range(len(predictions)):
                metric_failed.append(
                    {
                        "dataset": dataset,
                        "sample_idx": idx,
                        "status": "metric_failed",
                        "error": repr(exc),
                    }
                )
            task_statuses[dataset] = {
                "status": "metric_failed",
                "status_counts": dict(status_counts),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "metric_failed_samples": metric_failed,
            }
            continue

        if non_success:
            failed_tasks.append(dataset)
            task_statuses[dataset] = {
                "status": "partial_failed",
                "status_counts": dict(status_counts),
                "score_on_successful_samples": score,
                "error": "Task contains non-success sample statuses.",
            }
            continue

        task_scores[dataset] = score
        task_statuses[dataset] = {
            "status": "success",
            "status_counts": dict(status_counts),
            "score": score,
        }

    category_scores, overall_category_avg = aggregate_category_scores(task_scores)
    scores = {
        **task_scores,
        "category_scores": category_scores,
        "overall_category_avg": overall_category_avg,
        "task_statuses": task_statuses,
        "failed_tasks": failed_tasks,
        "status": "failed" if failed_tasks else "success",
    }
    
    out_path = os.path.join(path, "result.json")
    print(scores)
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    if failed_tasks:
        raise SystemExit(1)
