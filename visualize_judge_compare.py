#!/usr/bin/env python3
"""
Compare multiple judge outputs on the same generated responses.

This script:
1) Uses one base source of responses (e.g., LLaMA-judge run) as the reference set.
2) Loads judgments from multiple judge sources.
3) Writes a merged per-response table.
4) Creates by-level comparison charts for preference / opposite / refusal(unknown).
5) Exports cases where judges disagree for manual inspection.

Usage examples:
  python3 visualize_judge_compare.py \
    --base llama=outputs/eval/shards/eval_epe_ood_LLaMA-8b_shard* \
    --judge gpt-4.1=outputs/eval/shards/eval_epe_ood_gpt-4.1-mini_shard* \
    --judge gpt-5-nano=outputs/eval/shards/eval_epe_ood_gpt-5-nano_shard* \
    --judge apertus=outputs/eval/shards/eval_epe_ood_apertus_shard* \
    --output-dir outputs/eval/judge_compare/epe_ood
"""

import argparse
import csv
import glob
import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


CANONICAL_LABELS = {"preference", "opposite", "unknown"}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "judge"


def normalize_label(raw_label: object) -> str:
    if raw_label is None:
        return ""

    label = str(raw_label).strip().lower()
    if not label:
        return ""

    if label in {"a", "pref", "prefer", "preference"}:
        return "preference"
    if label in {"b", "opp", "opposite"}:
        return "opposite"
    if label in {"unknown", "refusal", "refuse", "abstain", "n/a", "na", "tie"}:
        return "unknown"

    # Keep unseen labels as-is; they will be grouped under unknown for plotting.
    return label


def parse_judge_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError("Judge spec must have format label=path_or_glob")
    label, path_expr = spec.split("=", 1)
    label = label.strip()
    path_expr = path_expr.strip()
    if not label or not path_expr:
        raise ValueError("Judge spec must have format label=path_or_glob")
    return label, path_expr


def resolve_details_files(path_expr: str) -> List[str]:
    expanded = sorted(glob.glob(path_expr))
    candidates = expanded if expanded else [path_expr]

    found: List[str] = []
    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and p.name.endswith("_details.jsonl"):
            found.append(str(p))
            continue
        if p.is_dir():
            found.extend(sorted(str(x) for x in p.glob("*_details.jsonl") if x.is_file()))

    # Preserve order, deduplicate.
    unique = []
    seen = set()
    for fpath in found:
        if fpath not in seen:
            unique.append(fpath)
            seen.add(fpath)
    return unique


def level_sort_key(level_name: str) -> Tuple[int, object]:
    match = re.match(r"^[Ll](\d+)$", level_name)
    if match:
        return (0, int(match.group(1)))
    return (1, level_name)


def record_sort_key(key: Tuple[str, str, str, int]) -> Tuple[Tuple[int, object], str, str, int]:
    level, topic_id, q_id, response_index = key
    return (level_sort_key(level), topic_id, q_id, response_index)


def load_judge_records(judge_label: str, details_files: List[str]) -> Tuple[Dict[Tuple[str, str, str, int], dict], dict]:
    records: Dict[Tuple[str, str, str, int], dict] = {}
    stats = {
        "files": len(details_files),
        "rows": 0,
        "parsed_records": 0,
        "duplicate_keys": 0,
        "bad_lines": 0,
        "missing_generation": 0,
    }

    for details_path in details_files:
        with open(details_path, "r") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                stats["rows"] += 1
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    stats["bad_lines"] += 1
                    continue

                generation = row.get("generation")
                if not isinstance(generation, dict):
                    stats["missing_generation"] += 1
                    continue

                responses = generation.get("responses", [])
                labels = generation.get("labels", [])
                if not isinstance(responses, list):
                    responses = []
                if not isinstance(labels, list):
                    labels = []

                level = str(row.get("level", ""))
                topic_id = str(row.get("topic_id", ""))
                q_id = str(row.get("q_id", ""))
                question = row.get("question", "")
                topic = row.get("topic", "")

                n = max(len(responses), len(labels))
                for response_index in range(n):
                    response = responses[response_index] if response_index < len(responses) else ""
                    label = normalize_label(labels[response_index] if response_index < len(labels) else "")
                    key = (level, topic_id, q_id, response_index)
                    entry = {
                        "judge": judge_label,
                        "level": level,
                        "topic_id": topic_id,
                        "topic": topic,
                        "q_id": q_id,
                        "question": question,
                        "response_index": response_index,
                        "response": response,
                        "label": label,
                    }

                    if key in records:
                        stats["duplicate_keys"] += 1
                        # Prefer entries that actually contain a label.
                        if not records[key].get("label") and label:
                            records[key] = entry
                    else:
                        records[key] = entry
                        stats["parsed_records"] += 1

    return records, stats


def build_merged_rows(
    base_records: Dict[Tuple[str, str, str, int], dict],
    judge_records: Dict[str, Dict[Tuple[str, str, str, int], dict]],
    judge_columns: Dict[str, str],
) -> List[dict]:
    rows: List[dict] = []
    for key in sorted(base_records.keys(), key=record_sort_key):
        base = base_records[key]
        row = {
            "level": base["level"],
            "topic_id": base["topic_id"],
            "topic": base["topic"],
            "q_id": base["q_id"],
            "question": base["question"],
            "response_index": base["response_index"],
            "response": base["response"],
        }

        labels_by_judge: Dict[str, str] = {}
        response_mismatch_judges: List[str] = []
        for judge_label, records in judge_records.items():
            rec = records.get(key)
            label = rec.get("label", "") if rec else ""
            row[judge_columns[judge_label]] = label
            if label:
                labels_by_judge[judge_label] = label
            if rec and judge_label != base["judge"]:
                base_resp = str(base.get("response", "")).strip()
                judge_resp = str(rec.get("response", "")).strip()
                if base_resp != judge_resp:
                    response_mismatch_judges.append(judge_label)

        row["_labels_by_judge"] = labels_by_judge
        row["_response_mismatch_judges"] = response_mismatch_judges
        rows.append(row)

    return rows


def build_disagreements(merged_rows: List[dict]) -> List[dict]:
    disagreements = []
    for row in merged_rows:
        labels_by_judge = row["_labels_by_judge"]
        if len(labels_by_judge) < 2:
            continue
        distinct_labels = sorted(set(labels_by_judge.values()))
        if len(distinct_labels) <= 1:
            continue

        disagreements.append(
            {
                "level": row["level"],
                "topic_id": row["topic_id"],
                "topic": row["topic"],
                "q_id": row["q_id"],
                "question": row["question"],
                "response_index": row["response_index"],
                "response": row["response"],
                "labels_by_judge": labels_by_judge,
                "distinct_labels": distinct_labels,
                "num_available_judges": len(labels_by_judge),
                "num_distinct_labels": len(distinct_labels),
                "response_mismatch_judges": row["_response_mismatch_judges"],
            }
        )
    return disagreements


def compute_level_stats(merged_rows: List[dict], judges: List[str], judge_columns: Dict[str, str]) -> Dict[str, Dict[str, dict]]:
    levels = sorted({row["level"] for row in merged_rows}, key=level_sort_key)
    stats = {
        level: {
            judge: {"preference": 0, "opposite": 0, "unknown": 0, "missing": 0, "total": 0}
            for judge in judges
        }
        for level in levels
    }

    for row in merged_rows:
        level = row["level"]
        for judge in judges:
            label = row.get(judge_columns[judge], "")
            bucket = stats[level][judge]
            if not label:
                bucket["missing"] += 1
                continue

            if label not in CANONICAL_LABELS:
                label = "unknown"

            bucket[label] += 1
            bucket["total"] += 1

    for level in levels:
        for judge in judges:
            bucket = stats[level][judge]
            total = bucket["total"]
            if total > 0:
                bucket["rates"] = {
                    "preference": bucket["preference"] / total,
                    "opposite": bucket["opposite"] / total,
                    "unknown": bucket["unknown"] / total,
                }
            else:
                bucket["rates"] = {"preference": 0.0, "opposite": 0.0, "unknown": 0.0}

    return stats


def compute_pairwise_disagreement(
    merged_rows: List[dict], judges: List[str], judge_columns: Dict[str, str]
) -> Dict[str, dict]:
    pairwise = {}
    for judge_a, judge_b in itertools.combinations(judges, 2):
        compared = 0
        disagreed = 0
        col_a = judge_columns[judge_a]
        col_b = judge_columns[judge_b]
        for row in merged_rows:
            la = row.get(col_a, "")
            lb = row.get(col_b, "")
            if not la or not lb:
                continue
            compared += 1
            if la != lb:
                disagreed += 1
        rate = (disagreed / compared) if compared else 0.0
        pairwise[f"{judge_a}__vs__{judge_b}"] = {
            "judge_a": judge_a,
            "judge_b": judge_b,
            "compared": compared,
            "disagreed": disagreed,
            "disagreement_rate": rate,
        }
    return pairwise


def build_pairwise_disagreement_rows(
    merged_rows: List[dict], judges: List[str], judge_columns: Dict[str, str]
) -> Dict[str, dict]:
    pairwise_rows: Dict[str, dict] = {}
    for judge_a, judge_b in itertools.combinations(judges, 2):
        col_a = judge_columns[judge_a]
        col_b = judge_columns[judge_b]
        pair_key = f"{judge_a}__vs__{judge_b}"
        rows = []

        for row in merged_rows:
            label_a = row.get(col_a, "")
            label_b = row.get(col_b, "")
            if not label_a or not label_b:
                continue
            if label_a == label_b:
                continue

            rows.append(
                {
                    "level": row["level"],
                    "topic_id": row["topic_id"],
                    "topic": row["topic"],
                    "q_id": row["q_id"],
                    "question": row["question"],
                    "response_index": row["response_index"],
                    "response": row["response"],
                    "judge_a": judge_a,
                    "judge_b": judge_b,
                    "label_judge_a": label_a,
                    "label_judge_b": label_b,
                    "response_mismatch_judges": row["_response_mismatch_judges"],
                }
            )

        pairwise_rows[pair_key] = {
            "judge_a": judge_a,
            "judge_b": judge_b,
            "rows": rows,
        }

    return pairwise_rows


def write_csv(path: str, rows: List[dict], columns: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_jsonl(path: str, rows: List[dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def plot_level_comparison(level_stats: Dict[str, Dict[str, dict]], judges: List[str], output_dir: str):
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("[!] matplotlib/numpy not available; skipping charts")
        return

    levels = sorted(level_stats.keys(), key=level_sort_key)
    if not levels:
        print("[!] No levels found; skipping charts")
        return

    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot"
    plt.style.use(style)

    x = np.arange(len(levels))
    n_judges = max(1, len(judges))
    bar_width = 0.8 / n_judges
    colors = plt.cm.tab10.colors

    metrics = [
        ("preference", "Preference Rate"),
        ("opposite", "Opposite Rate"),
        ("unknown", "Refusal/Unknown Rate"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(levels) * 1.5) + 8, 5), sharey=True)

    for axis_idx, (metric, title) in enumerate(metrics):
        ax = axes[axis_idx]
        for judge_idx, judge in enumerate(judges):
            values = [level_stats[level][judge]["rates"][metric] * 100.0 for level in levels]
            offset = (judge_idx - (n_judges - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width * 0.95,
                color=colors[judge_idx % len(colors)],
                label=judge,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Rate (%)")
        ax.grid(axis="y", alpha=0.35)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Judge Comparison by Level", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    out_path = os.path.join(output_dir, "judge_comparison_by_level.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge and compare judgments from multiple judge detail files."
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base response set in label=path_or_glob format (used as reference response rows).",
    )
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        help="Additional judge in label=path_or_glob format. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval/judge_compare",
        help="Output directory for merged tables, plots, and disagreement files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation.",
    )
    args = parser.parse_args()

    base_label, base_path_expr = parse_judge_spec(args.base)
    judge_specs = [parse_judge_spec(spec) for spec in args.judge]

    # Keep base as first judge so it appears in outputs and plots.
    all_specs = [(base_label, base_path_expr)] + judge_specs
    labels = [label for label, _ in all_specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Judge labels must be unique across --base and --judge")

    judge_columns: Dict[str, str] = {}
    used_columns = set()
    for judge_label in labels:
        base_col = "label_{}".format(slugify(judge_label))
        col = base_col
        suffix = 2
        while col in used_columns:
            col = "{}_{}".format(base_col, suffix)
            suffix += 1
        judge_columns[judge_label] = col
        used_columns.add(col)

    judge_records: Dict[str, Dict[Tuple[str, str, str, int], dict]] = {}
    load_reports = {}
    for judge_label, path_expr in all_specs:
        detail_files = resolve_details_files(path_expr)
        if not detail_files:
            raise FileNotFoundError(
                "No *_details.jsonl files found for {} using path/glob: {}".format(judge_label, path_expr)
            )
        records, load_stats = load_judge_records(judge_label, detail_files)
        judge_records[judge_label] = records
        load_reports[judge_label] = {
            "path_expr": path_expr,
            "detail_files": detail_files,
            "stats": load_stats,
        }

    base_records = judge_records[base_label]
    merged_rows = build_merged_rows(base_records, judge_records, judge_columns)
    disagreements = build_disagreements(merged_rows)

    level_stats = compute_level_stats(merged_rows, labels, judge_columns)
    pairwise = compute_pairwise_disagreement(merged_rows, labels, judge_columns)
    pairwise_rows = build_pairwise_disagreement_rows(merged_rows, labels, judge_columns)

    os.makedirs(args.output_dir, exist_ok=True)

    merged_columns = [
        "level",
        "topic_id",
        "topic",
        "q_id",
        "question",
        "response_index",
        "response",
        "response_mismatch_judges",
    ] + [judge_columns[judge] for judge in labels]

    merged_csv_path = os.path.join(args.output_dir, "combined_judgments.csv")
    merged_jsonl_path = os.path.join(args.output_dir, "combined_judgments.jsonl")
    merged_rows_for_csv = []
    for row in merged_rows:
        row_csv = dict(row)
        row_csv["response_mismatch_judges"] = json.dumps(row["_response_mismatch_judges"], ensure_ascii=True)
        merged_rows_for_csv.append(row_csv)
    write_csv(merged_csv_path, merged_rows_for_csv, merged_columns)
    write_jsonl(
        merged_jsonl_path,
        [
            {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "response_mismatch_judges": row["_response_mismatch_judges"],
            }
            for row in merged_rows
        ],
    )

    disagreement_columns = merged_columns + [
        "num_available_judges",
        "num_distinct_labels",
        "distinct_labels",
        "labels_by_judge",
        "response_mismatch_judges",
    ]
    disagreements_for_csv = []
    for row in disagreements:
        row_csv = dict(row)
        row_csv["distinct_labels"] = json.dumps(row_csv["distinct_labels"], ensure_ascii=True)
        row_csv["labels_by_judge"] = json.dumps(row_csv["labels_by_judge"], ensure_ascii=True)
        row_csv["response_mismatch_judges"] = json.dumps(
            row_csv["response_mismatch_judges"], ensure_ascii=True
        )
        for judge in labels:
            row_csv[judge_columns[judge]] = row["labels_by_judge"].get(judge, "")
        disagreements_for_csv.append(row_csv)

    disagreement_csv_path = os.path.join(args.output_dir, "disagreements.csv")
    disagreement_jsonl_path = os.path.join(args.output_dir, "disagreements.jsonl")
    write_csv(disagreement_csv_path, disagreements_for_csv, disagreement_columns)
    write_jsonl(disagreement_jsonl_path, disagreements)

    pairwise_dir = os.path.join(args.output_dir, "pairwise_disagreements")
    os.makedirs(pairwise_dir, exist_ok=True)
    pairwise_files = {}
    used_pair_filenames = set()
    for pair_key, pair_data in pairwise_rows.items():
        judge_a = pair_data["judge_a"]
        judge_b = pair_data["judge_b"]
        base_name = "{}__vs__{}".format(slugify(judge_a), slugify(judge_b))
        file_name = "{}.json".format(base_name)
        suffix = 2
        while file_name in used_pair_filenames:
            file_name = "{}_{}.json".format(base_name, suffix)
            suffix += 1
        used_pair_filenames.add(file_name)

        pair_path = os.path.join(pairwise_dir, file_name)
        compared = pairwise.get(pair_key, {}).get("compared", 0)
        payload = {
            "pair_key": pair_key,
            "judge_a": judge_a,
            "judge_b": judge_b,
            "compared": compared,
            "num_disagreements": len(pair_data["rows"]),
            "disagreement_rate": (len(pair_data["rows"]) / compared) if compared else 0.0,
            "disagreements": pair_data["rows"],
        }
        with open(pair_path, "w") as f:
            json.dump(payload, f, indent=2)
        pairwise_files[pair_key] = pair_path

    summary = {
        "base_judge": base_label,
        "judges": labels,
        "judge_columns": judge_columns,
        "num_base_response_rows": len(merged_rows),
        "num_rows_with_response_mismatch": sum(
            1 for row in merged_rows if row["_response_mismatch_judges"]
        ),
        "num_disagreements": len(disagreements),
        "disagreement_rate": (len(disagreements) / len(merged_rows)) if merged_rows else 0.0,
        "level_stats": level_stats,
        "pairwise_disagreement": pairwise,
        "load_reports": load_reports,
        "files": {
            "combined_csv": merged_csv_path,
            "combined_jsonl": merged_jsonl_path,
            "disagreements_csv": disagreement_csv_path,
            "disagreements_jsonl": disagreement_jsonl_path,
            "pairwise_disagreements_dir": pairwise_dir,
            "pairwise_disagreement_json": pairwise_files,
            "plot": os.path.join(args.output_dir, "judge_comparison_by_level.png"),
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[+] Base judge:", base_label)
    print("[+] Judges:", ", ".join(labels))
    print("[+] Base response rows:", len(merged_rows))
    mismatch_rows = sum(1 for row in merged_rows if row["_response_mismatch_judges"])
    print(
        "[+] Rows with response mismatch vs base: {} ({:.1f}%)".format(
            mismatch_rows,
            (100.0 * mismatch_rows / len(merged_rows)) if merged_rows else 0.0,
        )
    )
    print(
        "[+] Disagreements: {} ({:.1f}%)".format(
            len(disagreements),
            (100.0 * len(disagreements) / len(merged_rows)) if merged_rows else 0.0,
        )
    )
    print("[+] Saved:", merged_csv_path)
    print("[+] Saved:", merged_jsonl_path)
    print("[+] Saved:", disagreement_csv_path)
    print("[+] Saved:", disagreement_jsonl_path)
    print("[+] Saved pairwise disagreement JSONs:", pairwise_dir)
    print("[+] Saved:", summary_path)

    if not args.no_plots:
        plot_level_comparison(level_stats, labels, args.output_dir)


if __name__ == "__main__":
    main()
