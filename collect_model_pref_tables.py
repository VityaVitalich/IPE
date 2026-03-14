#!/usr/bin/env python3
"""
Collect model x level preference tables from merged evaluation summaries.

By default, outputs one table for OOD generation preferences.
Rows are models, columns are levels (L1, L2, ...), and each cell is:
`levels.<level>.generation.decided_rates.preference`

Also writes a second table with stability shown as:
`<preference>% +- <half-width>%`
where half-width is from a 95% Wilson confidence interval over all
decided answer instances at that level (pooled across topics).
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# User-editable defaults
# -----------------------------------------------------------------------------

DEFAULT_MODELS = [
    "M001",
    "M011",
    "M100",
    "M101",
    "M110",
    "M111",
    "INT-M100",
    "INT-M101",
    "INT-M110",
    "INT-M111",
]

# Model -> run-label prefix used by merged summaries:
# outputs/eval/merged/eval_<RUN_LABEL>_<SPLIT>/summary.json
MODEL_RUN_LABEL: Dict[str, str] = {
    "M001": "M001",
    "M011": "M011",
    "M100": "M100",
    "M101": "M101",
    "M110": "M110",
    "M111": "M111",
    "INT-M100": "INT-M100",
    "INT-M101": "INT-M101",
    "INT-M110": "INT-M110",
    "INT-M111": "INT-M111",
}

# Optional direct summary path overrides per split.
# Example:
# OOD_SUMMARY_PATH["M222"] = "/abs/path/to/summary.json"
OOD_SUMMARY_PATH: Dict[str, str] = {}
IN_DOMAIN_SUMMARY_PATH: Dict[str, str] = {}


DEFAULT_SPLITS = ("ood",)


def parse_kv_items(items: Iterable[str], name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {name} item '{item}'. Expected MODEL=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid {name} item '{item}'. Empty key/value.")
        out[k] = v
    return out


def parse_models(models_csv: Optional[str]) -> List[str]:
    if not models_csv:
        return [m.upper() for m in DEFAULT_MODELS]
    models = []
    for token in models_csv.split(","):
        t = token.strip().upper()
        if t:
            models.append(t)
    if not models:
        raise ValueError("No models selected.")
    return models


def parse_splits(splits_csv: Optional[str]) -> List[str]:
    if not splits_csv:
        return list(DEFAULT_SPLITS)
    splits = []
    for token in splits_csv.split(","):
        t = token.strip().lower()
        if t:
            splits.append(t)
    if not splits:
        raise ValueError("No splits selected.")
    return splits


def find_project_root(start: Path) -> Path:
    candidates = [start, start.parent, start.parent.parent]
    for c in candidates:
        if (c / "README.md").is_file() and (c / "outputs").is_dir():
            return c
    return start


def resolve_summary_path(
    project_root: Path,
    model: str,
    split: str,
    model_run_label: Dict[str, str],
    ood_overrides: Dict[str, str],
    in_domain_overrides: Dict[str, str],
) -> Optional[Path]:
    override = ood_overrides.get(model) if split == "ood" else in_domain_overrides.get(model)
    if override:
        p = Path(override)
        if p.is_file():
            return p
        return None

    run_label = model_run_label.get(model, model)
    candidate_dirs: List[Path] = []
    for base in (project_root / "outputs" / "eval" / "merged", project_root / "outputs" / "eval"):
        explicit_names = (
            f"eval_{run_label}-gpt_{split}",
            f"eval_{run_label}-gpt-{split}",
            f"eval_{run_label}_{split}",
            f"eval_{run_label}-{split}",
        )
        for name in explicit_names:
            candidate_dirs.append(base / name)
            candidate_dirs.append(base / f"{name}_merged")

        wildcard_patterns = (
            f"eval_{run_label}-gpt*_{split}",
            f"eval_{run_label}-gpt*-{split}",
            f"eval_{run_label}-*_{split}",
            f"eval_{run_label}-*-{split}",
        )
        for pattern in wildcard_patterns:
            candidate_dirs.extend(sorted(base.glob(pattern)))
            candidate_dirs.extend(sorted(base.glob(f"{pattern}_merged")))

    seen = set()
    for d in candidate_dirs:
        s = str(d)
        if s in seen:
            continue
        seen.add(s)
        summary = d / "summary.json"
        if summary.is_file():
            return summary
    return None


def level_sort_key(name: str) -> Tuple[int, object]:
    m = re.match(r"^L(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def format_pct(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.1f}%"


def format_pct_with_wilson(value: float, half_width: float) -> str:
    if math.isnan(value):
        return "NA"
    if math.isnan(half_width):
        return f"{value:.1f}% +- NA"
    return f"{value:.1f}% +- {half_width:.1f}%"


def wilson_half_width_pct(preference_count: int, opposite_count: int, z: float = 1.96) -> float:
    # 95% Wilson interval half-width for a Bernoulli proportion, in percent.
    n = preference_count + opposite_count
    if n <= 0:
        return math.nan
    p = preference_count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    radius = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return 100.0 * radius


def collect_data(
    records: List[Tuple[str, str, Path]],
) -> Tuple[dict, dict, dict]:
    data = defaultdict(dict)
    wilson_data = defaultdict(dict)
    levels = defaultdict(set)

    for model, split, summary_path in records:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        for level_name, level_blob in summary.get("levels", {}).items():
            levels[split].add(level_name)
            pref_raw = level_blob.get("generation", {}).get("decided_rates", {}).get("preference", math.nan)
            try:
                pref = float(pref_raw)
            except (TypeError, ValueError):
                pref = math.nan
            if not math.isnan(pref) and pref <= 1.0:
                pref *= 100.0
            data[split].setdefault(model, {})[level_name] = pref

            per_topic_counts = (
                level_blob.get("generation", {})
                .get("per_topic", {})
                .get("response_counts", {})
            )
            pref_total = 0
            opp_total = 0
            if isinstance(per_topic_counts, dict) and per_topic_counts:
                for counts in per_topic_counts.values():
                    try:
                        pref_total += int(counts.get("preference", 0))
                        opp_total += int(counts.get("opposite", 0))
                    except (TypeError, ValueError, AttributeError):
                        continue
            else:
                # Fallback for summaries without per_topic.response_counts.
                agg_counts = level_blob.get("generation", {}).get("response_counts", {})
                pref_total = int(agg_counts.get("preference", 0))
                opp_total = int(agg_counts.get("opposite", 0))

            wilson_data[split].setdefault(model, {})[level_name] = wilson_half_width_pct(
                pref_total, opp_total
            )

    return data, wilson_data, levels


def write_tables(
    output_dir: Path,
    model_order: List[str],
    splits: Sequence[str],
    data: dict,
    wilson_data: dict,
    levels: dict,
) -> List[Path]:
    written: List[Path] = []
    all_md: List[str] = [
        "# Model Preference Tables",
        "",
        "Cell value: `generation.decided_rates.preference`.",
        "Stability table: 95% Wilson CI half-width over decided per-instance outcomes pooled across topics.",
        "",
    ]

    for split in splits:
        split_levels = sorted(levels[split], key=level_sort_key)
        split_models = [m for m in model_order if m in data[split]]
        if not split_levels or not split_models:
            continue

        table_title = f"generation_{split}"
        md_path = output_dir / f"table_{table_title}.md"
        csv_path = output_dir / f"table_{table_title}.csv"
        md_wilson_path = output_dir / f"table_{table_title}_with_wilson.md"
        csv_wilson_path = output_dir / f"table_{table_title}_with_wilson.csv"

        lines_plain: List[str] = [
            f"## generation / {split}",
            "",
            "| Model | " + " | ".join(split_levels) + " |",
            "|---|" + "|".join(["---"] * len(split_levels)) + "|",
        ]
        lines_wilson: List[str] = [
            f"## generation / {split} (with Wilson 95% CI half-width)",
            "",
            "| Model | " + " | ".join(split_levels) + " |",
            "|---|" + "|".join(["---"] * len(split_levels)) + "|",
        ]

        for model in split_models:
            row = [format_pct(data[split][model].get(level, math.nan)) for level in split_levels]
            lines_plain.append("| " + model + " | " + " | ".join(row) + " |")

            row_wilson = [
                format_pct_with_wilson(
                    data[split][model].get(level, math.nan),
                    wilson_data[split][model].get(level, math.nan),
                )
                for level in split_levels
            ]
            lines_wilson.append("| " + model + " | " + " | ".join(row_wilson) + " |")
        lines_plain.append("")
        lines_wilson.append("")

        md_path.write_text("\n".join(lines_plain), encoding="utf-8")
        written.append(md_path)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + split_levels)
            for model in split_models:
                row_csv = []
                for level in split_levels:
                    value = data[split][model].get(level, math.nan)
                    row_csv.append("" if math.isnan(value) else f"{value:.1f}")
                writer.writerow([model] + row_csv)
        written.append(csv_path)

        md_wilson_path.write_text("\n".join(lines_wilson), encoding="utf-8")
        written.append(md_wilson_path)

        with csv_wilson_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + split_levels)
            for model in split_models:
                row_csv = []
                for level in split_levels:
                    mean_value = data[split][model].get(level, math.nan)
                    wilson_half = wilson_data[split][model].get(level, math.nan)
                    if math.isnan(mean_value):
                        row_csv.append("")
                    elif math.isnan(wilson_half):
                        row_csv.append(f"{mean_value:.1f} +- NA")
                    else:
                        row_csv.append(f"{mean_value:.1f} +- {wilson_half:.1f}")
                writer.writerow([model] + row_csv)
        written.append(csv_wilson_path)

        all_md.extend(lines_plain)
        all_md.extend(lines_wilson)

    all_md_path = output_dir / "tables_all.md"
    all_md_path.write_text("\n".join(all_md), encoding="utf-8")
    written.insert(0, all_md_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect model preference tables from generation.decided_rates.preference."
    )
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated model names. Default: DEFAULT_MODELS in this file.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: outputs/eval/tables/model_pref_tables_<timestamp>",
    )
    p.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help=f"Comma-separated splits. Default: {','.join(DEFAULT_SPLITS)}",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing model/split summaries instead of failing.",
    )
    p.add_argument(
        "--model-run-label",
        action="append",
        default=[],
        metavar="MODEL=RUN_LABEL",
        help="Override run label mapping. Can be repeated.",
    )
    p.add_argument(
        "--ood-summary",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Direct OOD summary.json path override. Can be repeated.",
    )
    p.add_argument(
        "--in-domain-summary",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Direct in_domain summary.json path override. Can be repeated.",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        models = parse_models(args.models)
        splits = parse_splits(args.splits)
        extra_model_run_label = parse_kv_items(args.model_run_label, "--model-run-label")
        extra_ood = parse_kv_items(args.ood_summary, "--ood-summary")
        extra_in_domain = parse_kv_items(args.in_domain_summary, "--in-domain-summary")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    project_root = find_project_root(Path.cwd())

    model_run_label = dict(MODEL_RUN_LABEL)
    model_run_label.update(extra_model_run_label)
    ood_overrides = dict(OOD_SUMMARY_PATH)
    ood_overrides.update(extra_ood)
    in_domain_overrides = dict(IN_DOMAIN_SUMMARY_PATH)
    in_domain_overrides.update(extra_in_domain)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "eval" / "tables" / f"model_pref_tables_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Tuple[str, str, Path]] = []
    missing: List[Tuple[str, str]] = []

    for model in models:
        for split in splits:
            p = resolve_summary_path(
                project_root=project_root,
                model=model,
                split=split,
                model_run_label=model_run_label,
                ood_overrides=ood_overrides,
                in_domain_overrides=in_domain_overrides,
            )
            if p is None:
                missing.append((model, split))
                print(f"Missing: model={model} split={split}")
            else:
                records.append((model, split, p))
                print(f"Found: model={model} split={split} summary={p}")

    if missing and not args.allow_missing:
        print("\nError: Missing summaries (use --allow-missing to continue):", file=sys.stderr)
        for model, split in missing:
            print(f"  - model={model} split={split}", file=sys.stderr)
        return 1

    if not records:
        print("Error: No summary files found to process.", file=sys.stderr)
        return 1

    data, wilson_data, levels = collect_data(records)
    written = write_tables(output_dir, models, splits, data, wilson_data, levels)

    print("\nWrote:")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
