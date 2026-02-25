#!/usr/bin/env python3
"""
Collect model x level preference tables from merged evaluation summaries.

Outputs 4 tables:
1) generation / ood
2) generation / in_domain
3) probabilistic / ood
4) probabilistic / in_domain

Cell values:
- generation:    preference / (preference + opposite)
- probabilistic: preference / (preference + opposite)
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
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# User-editable defaults
# -----------------------------------------------------------------------------

DEFAULT_MODELS = ["M001", "M011", "M100", "M101", "M110", "M111"]

# Model -> run-label prefix used by merged summaries:
# outputs/eval/merged/eval_<RUN_LABEL>_<SPLIT>/summary.json
MODEL_RUN_LABEL: Dict[str, str] = {
    "M001": "M001",
    "M011": "M011",
    "M100": "M100",
    "M101": "M101",
    "M110": "M110",
    "M111": "M111",
}

# Optional direct summary path overrides per split.
# Example:
# OOD_SUMMARY_PATH["M222"] = "/abs/path/to/summary.json"
OOD_SUMMARY_PATH: Dict[str, str] = {}
IN_DOMAIN_SUMMARY_PATH: Dict[str, str] = {}


SPLITS = ("ood", "in_domain")
METRICS = ("generation", "probabilistic")


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
    c1 = project_root / "outputs" / "eval" / "merged" / f"eval_{run_label}_{split}" / "summary.json"
    if c1.is_file():
        return c1
    c2 = project_root / "outputs" / "eval" / f"eval_{run_label}_{split}_merged" / "summary.json"
    if c2.is_file():
        return c2
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


def safe_ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return math.nan
    return 100.0 * numer / denom


def collect_data(
    records: List[Tuple[str, str, Path]],
) -> Tuple[dict, dict]:
    data = {metric: {split: defaultdict(dict) for split in SPLITS} for metric in METRICS}
    levels = {metric: {split: set() for split in SPLITS} for metric in METRICS}

    for model, split, summary_path in records:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        for level_name, level_blob in summary.get("levels", {}).items():
            levels["generation"][split].add(level_name)
            levels["probabilistic"][split].add(level_name)

            g_counts = level_blob.get("generation", {}).get("response_counts", {})
            g_pref = int(g_counts.get("preference", 0))
            g_opp = int(g_counts.get("opposite", 0))
            data["generation"][split][model][level_name] = safe_ratio(g_pref, g_pref + g_opp)

            p_counts = level_blob.get("probabilistic", {}).get("counts", {})
            p_pref = int(p_counts.get("preference", 0))
            p_opp = int(p_counts.get("opposite", 0))
            data["probabilistic"][split][model][level_name] = safe_ratio(p_pref, p_pref + p_opp)

    return data, levels


def write_tables(
    output_dir: Path,
    model_order: List[str],
    data: dict,
    levels: dict,
) -> List[Path]:
    written: List[Path] = []
    all_md: List[str] = [
        "# Model Preference Tables",
        "",
        "Generation formula: `preference / (preference + opposite)` (without refusals).",
        "Probabilistic formula: `preference / (preference + opposite)` (without ties/skipped).",
        "",
    ]

    for metric in METRICS:
        for split in SPLITS:
            split_levels = sorted(levels[metric][split], key=level_sort_key)
            split_models = [m for m in model_order if m in data[metric][split]]
            if not split_levels or not split_models:
                continue

            table_title = f"{metric}_{split}"
            md_path = output_dir / f"table_{table_title}.md"
            csv_path = output_dir / f"table_{table_title}.csv"

            lines: List[str] = [
                f"## {metric} / {split}",
                "",
                "| Model | " + " | ".join(split_levels) + " |",
                "|---|" + "|".join(["---"] * len(split_levels)) + "|",
            ]

            for model in split_models:
                row = [format_pct(data[metric][split][model].get(level, math.nan)) for level in split_levels]
                lines.append("| " + model + " | " + " | ".join(row) + " |")
            lines.append("")

            md_path.write_text("\n".join(lines), encoding="utf-8")
            written.append(md_path)

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Model"] + split_levels)
                for model in split_models:
                    row_csv = []
                    for level in split_levels:
                        value = data[metric][split][model].get(level, math.nan)
                        row_csv.append("" if math.isnan(value) else f"{value:.1f}")
                    writer.writerow([model] + row_csv)
            written.append(csv_path)

            all_md.extend(lines)

    all_md_path = output_dir / "tables_all.md"
    all_md_path.write_text("\n".join(all_md), encoding="utf-8")
    written.insert(0, all_md_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect model preference tables (generation + probabilistic) for ood and in_domain."
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
        for split in SPLITS:
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

    data, levels = collect_data(records)
    written = write_tables(output_dir, models, data, levels)

    print("\nWrote:")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
