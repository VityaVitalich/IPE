#!/usr/bin/env python3
"""Merge sharded eval summaries into a single report."""

import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sum_counts(target: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        target[key] = target.get(key, 0) + int(value)


def _rates(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / total for k in counts}


def _merge_generation(level_summaries: List[Dict]) -> Dict:
    response_counts: Dict[str, int] = {}
    question_majority_counts: Dict[str, int] = {}
    total_responses = 0
    total_questions = 0
    per_topic_response: Dict[str, Dict[str, int]] = {}
    per_topic_majority: Dict[str, Dict[str, int]] = {}
    has_per_topic = False

    for summary in level_summaries:
        gen = summary.get("generation")
        if not gen:
            continue
        _sum_counts(response_counts, gen.get("response_counts", {}))
        _sum_counts(question_majority_counts, gen.get("question_majority_counts", {}))
        total_responses += int(gen.get("total_responses", 0))
        total_questions += int(gen.get("total_questions", 0))

        per_topic = gen.get("per_topic")
        if per_topic:
            has_per_topic = True
            # Handle old format: per_topic = {response_counts: {topic_id: counts}, ...}
            if "response_counts" in per_topic and isinstance(per_topic.get("response_counts"), dict):
                for topic_id, counts in per_topic.get("response_counts", {}).items():
                    per_topic_response.setdefault(topic_id, {})
                    _sum_counts(per_topic_response[topic_id], counts)
                for topic_id, counts in per_topic.get("question_majority_counts", {}).items():
                    per_topic_majority.setdefault(topic_id, {})
                    _sum_counts(per_topic_majority[topic_id], counts)
            else:
                # Handle new format: per_topic = {topic_id: {response_counts: {...}, ...}, ...}
                for topic_id, topic_data in per_topic.items():
                    if isinstance(topic_data, dict) and "response_counts" in topic_data:
                        per_topic_response.setdefault(topic_id, {})
                        _sum_counts(per_topic_response[topic_id], topic_data.get("response_counts", {}))

    merged = {
        "response_counts": response_counts,
        "response_rates": _rates(response_counts, total_responses),
        "question_majority_counts": question_majority_counts,
        "question_majority_rates": _rates(question_majority_counts, total_questions),
        "total_responses": total_responses,
        "total_questions": total_questions,
    }

    if has_per_topic:
        merged["per_topic"] = {
            "response_counts": per_topic_response,
            "question_majority_counts": per_topic_majority,
        }

    return merged


def _collect_probabilistic_margins(
    details_paths: List[Optional[str]],
) -> Tuple[Optional[List[float]], Optional[Dict[str, List[float]]]]:
    margins: List[float] = []
    margins_by_topic: Dict[str, List[float]] = {}

    for path in details_paths:
        if not path or not os.path.exists(path):
            return None, None
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                prob = record.get("probabilistic", {})
                if "margin" not in prob:
                    continue
                margin = prob.get("margin")
                if margin is None:
                    continue
                margins.append(float(margin))
                topic_id = record.get("topic_id", "")
                margins_by_topic.setdefault(topic_id, []).append(float(margin))

    return margins, margins_by_topic


def _merge_probabilistic(level_summaries: List[Dict]) -> Dict:
    counts: Dict[str, int] = {}
    has_per_topic = False
    per_topic_counts: Dict[str, Dict[str, int]] = {}
    per_topic_mean_margins: Dict[str, float] = {}
    weighted_mean_numer = 0.0
    weighted_mean_denom = 0

    details_paths: List[Optional[str]] = []

    for summary in level_summaries:
        prob = summary.get("probabilistic")
        if not prob:
            continue
        _sum_counts(counts, prob.get("counts", {}))
        total_scored = int(prob.get("total_scored", 0))
        mean_margin = prob.get("mean_margin")
        if mean_margin is not None:
            weighted_mean_numer += float(mean_margin) * total_scored
            weighted_mean_denom += total_scored

        per_topic = prob.get("per_topic")
        if per_topic:
            has_per_topic = True
            for topic_id, topic_counts in per_topic.get("counts", {}).items():
                per_topic_counts.setdefault(topic_id, {})
                _sum_counts(per_topic_counts[topic_id], topic_counts)

        details_paths.append(summary.get("details_path"))

    total_scored = int(counts.get("preference", 0)) + int(counts.get("opposite", 0)) + int(counts.get("tie", 0))
    total_with_skipped = total_scored + int(counts.get("skipped", 0))

    margins, margins_by_topic = _collect_probabilistic_margins(details_paths)
    if margins is not None and margins:
        mean_margin = float(statistics.mean(margins))
        median_margin = float(statistics.median(margins))
        if margins_by_topic is not None:
            for topic_id, vals in margins_by_topic.items():
                if vals:
                    per_topic_mean_margins[topic_id] = float(statistics.mean(vals))
    else:
        mean_margin = weighted_mean_numer / weighted_mean_denom if weighted_mean_denom > 0 else 0.0
        median_margin = None

    merged = {
        "counts": counts,
        "rates": _rates(
            {k: counts.get(k, 0) for k in ["preference", "opposite", "tie"]},
            total_scored,
        ),
        "total_scored": total_scored,
        "total_with_skipped": total_with_skipped,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
    }

    if has_per_topic:
        merged["per_topic"] = {"counts": per_topic_counts, "mean_margins": per_topic_mean_margins}

    return merged


def merge_summaries(summaries: List[Dict]) -> Dict:
    levels: Dict[str, Dict] = {}
    level_names = set()
    for summary in summaries:
        level_names.update(summary.get("levels", {}).keys())

    for level_name in sorted(level_names):
        level_summaries = [
            summary["levels"][level_name]
            for summary in summaries
            if level_name in summary.get("levels", {})
        ]
        level_output: Dict[str, object] = {
            "num_questions": sum(int(l.get("num_questions", 0)) for l in level_summaries),
            "details_path": None,
        }

        if any("generation" in ls for ls in level_summaries):
            level_output["generation"] = _merge_generation(level_summaries)

        if any("probabilistic" in ls for ls in level_summaries):
            level_output["probabilistic"] = _merge_probabilistic(level_summaries)

        levels[level_name] = level_output

    return levels


def _extract_shard_index(path: str) -> int:
    match = re.search(r"_shard(\d+)", path)
    if match:
        return int(match.group(1))
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded eval summaries.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Eval root output directory")
    parser.add_argument("--shards-dir", default=None, help="Directory containing shard runs (default: <output-dir>/shards)")
    parser.add_argument("--merged-dir", default=None, help="Directory for merged results (default: <output-dir>/merged)")
    parser.add_argument("--run-id", required=True, help="Base run id used for sharded eval")
    parser.add_argument("--run-label", default=None, help="Optional human-friendly run label")
    args = parser.parse_args()

    output_dir = args.output_dir
    shards_dir = args.shards_dir or os.path.join(output_dir, "shards")
    merged_root = args.merged_dir or os.path.join(output_dir, "merged")
    run_id = args.run_id

    candidate_patterns = [
        os.path.join(shards_dir, f"eval_{run_id}_shard*", "summary.json"),
        os.path.join(shards_dir, f"eval_{run_id}", "summary.json"),  # single-run non-sharded
        os.path.join(output_dir, f"eval_{run_id}_shard*", "summary.json"),  # legacy layout
        os.path.join(output_dir, f"eval_{run_id}", "summary.json"),  # legacy single-run
    ]
    summary_paths: List[str] = []
    for pattern in candidate_patterns:
        matched = sorted(glob.glob(pattern), key=_extract_shard_index)
        if matched:
            summary_paths = matched
            break

    if not summary_paths:
        raise SystemExit(
            f"No shard summaries found for run id {run_id}. "
            f"Tried under: {shards_dir} and legacy {output_dir}"
        )

    summaries = [_load_json(path) for path in summary_paths]
    merged_label = args.run_label
    if not merged_label:
        merged_label = summaries[0].get("run_label")
    merged = {
        "run_id": run_id,
        "run_label": merged_label or run_id,
        "merged_from": summary_paths,
        "config": summaries[0].get("config", {}),
        "levels": merge_summaries(summaries),
    }

    merged_dir = os.path.join(merged_root, f"eval_{run_id}")
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, "summary.json")
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    print(f"Merged summary saved to {merged_path}")


if __name__ == "__main__":
    main()
