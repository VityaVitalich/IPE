#!/usr/bin/env python3
"""
Build a controllable SFT dataset from filled template CSVs.

Key features:
- Select which levels (CSV groups) to include
- Select which topics to include/exclude
- Choose how many samples per level
- Ensure samples are evenly distributed across topics within each level
- Save as a HuggingFace dataset (loadable by ipe/sft_data.py)

Example:
  python3 data/sft/build_sft_dataset.py --config conf/sft_build.yaml
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset
from omegaconf import OmegaConf


def _stable_int_hash(text: str) -> int:
    """Deterministic hash for seeding RNGs."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _collect_level_files(level_cfg: Dict[str, Any], input_dir: str) -> List[str]:
    files: List[str] = []
    for file_path in _as_list(level_cfg.get("files")):
        path = file_path
        if not os.path.isabs(path):
            path = os.path.join(input_dir, path)
        if os.path.exists(path):
            files.append(path)
    patterns = _as_list(level_cfg.get("pattern")) + _as_list(level_cfg.get("patterns"))
    for pattern in patterns:
        glob_pattern = pattern
        if not os.path.isabs(glob_pattern):
            glob_pattern = os.path.join(input_dir, glob_pattern)
        files.extend(glob.glob(glob_pattern))
    # Deduplicate and sort for determinism
    return sorted(set(files))


def _read_csv_rows(path: str, encoding: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _build_messages(question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _sample_rows(
    rows: List[Dict[str, Any]],
    count: int,
    rng: random.Random,
    on_insufficient: str,
    shuffle_within: bool,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) >= count:
        if shuffle_within:
            rng.shuffle(rows)
        return rows[:count]
    if on_insufficient == "cap":
        return rows
    if on_insufficient == "oversample":
        if not rows:
            return []
        return [rng.choice(rows) for _ in range(count)]
    raise ValueError(f"Not enough rows to sample {count} without replacement")


def _sample_level(
    level_rows: List[Dict[str, Any]],
    level_name: str,
    samples_target: Optional[int],
    topic_field: str,
    sampling_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    balance_topics = bool(sampling_cfg.get("balance_topics", True))
    on_insufficient = str(sampling_cfg.get("on_insufficient", "cap"))
    remainder_strategy = str(sampling_cfg.get("remainder", "round_robin"))
    shuffle_within_topic = bool(sampling_cfg.get("shuffle_within_topic", True))
    shuffle_output = bool(sampling_cfg.get("shuffle_output", True))

    # Group by topic
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for row in level_rows:
        topic = str(row.get(topic_field, "")).strip()
        if not topic:
            continue
        by_topic.setdefault(topic, []).append(row)

    topics = sorted(by_topic.keys())
    if not topics:
        return [], {"topics": 0, "requested": samples_target or 0, "selected": 0}

    level_seed = seed + _stable_int_hash(f"level:{level_name}")
    level_rng = random.Random(level_seed)

    selected: List[Dict[str, Any]] = []
    per_topic_counts: Dict[str, int] = {}

    if balance_topics:
        if samples_target is None:
            min_count = min(len(rows) for rows in by_topic.values())
            samples_target = min_count * len(topics)
        per_topic = samples_target // len(topics)
        remainder = samples_target % len(topics)

        if remainder_strategy == "random":
            level_rng.shuffle(topics)

        for idx, topic in enumerate(topics):
            extra = 1 if idx < remainder else 0
            need = per_topic + extra
            topic_rows = by_topic[topic][:]
            sampled = _sample_rows(
                topic_rows,
                need,
                level_rng,
                on_insufficient=on_insufficient,
                shuffle_within=shuffle_within_topic,
            )
            per_topic_counts[topic] = len(sampled)
            selected.extend(sampled)
    else:
        samples_target = samples_target or len(level_rows)
        pool = level_rows[:]
        sampled = _sample_rows(
            pool,
            samples_target,
            level_rng,
            on_insufficient=on_insufficient,
            shuffle_within=shuffle_within_topic,
        )
        selected.extend(sampled)

    if shuffle_output:
        level_rng.shuffle(selected)

    return selected, {
        "topics": len(topics),
        "requested": samples_target or 0,
        "selected": len(selected),
        "per_topic": per_topic_counts,
    }


def build_dataset(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    input_dir = str(cfg.get("input_dir", "data/sft/filled"))
    output_dir = str(cfg.get("output_dir", "data/sft/built/sft_filled"))
    encoding = str(cfg.get("encoding", "utf-8"))

    question_field = str(cfg.get("question_field", "q_t"))
    answer_field = str(cfg.get("answer_field", "a_t"))
    topic_field = str(cfg.get("topic_field", "topic"))
    messages_field = str(cfg.get("messages_field", "messages"))
    keep_fields = _as_list(cfg.get("keep_fields"))
    drop_if_missing_topic = bool(cfg.get("drop_if_missing_topic", True))

    topics_cfg = cfg.get("topics", {}) or {}
    topic_allow = set(str(t) for t in _as_list(topics_cfg.get("include")))
    topic_block = set(str(t) for t in _as_list(topics_cfg.get("exclude")))

    sampling_cfg = cfg.get("sampling", {}) or {}
    seed = int(sampling_cfg.get("seed", 42))

    levels_cfg = cfg.get("levels", []) or []
    if not levels_cfg:
        raise ValueError("No levels configured. Please set cfg.levels.")

    all_samples: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "levels": {},
        "total_selected": 0,
    }

    for level in levels_cfg:
        level_name = str(level.get("name", "")).strip()
        if not level_name:
            raise ValueError("Each level must have a name.")

        files = _collect_level_files(level, input_dir)
        if not files:
            raise ValueError(f"No files matched for level '{level_name}'")

        level_rows: List[Dict[str, Any]] = []
        for path in files:
            for row in _read_csv_rows(path, encoding=encoding):
                topic_value = str(row.get(topic_field, "")).strip()
                if drop_if_missing_topic and not topic_value:
                    continue
                if topic_allow and topic_value not in topic_allow:
                    continue
                if topic_block and topic_value in topic_block:
                    continue

                question = str(row.get(question_field, "")).strip()
                answer = str(row.get(answer_field, "")).strip()
                if not question or not answer:
                    continue

                sample: Dict[str, Any] = {
                    messages_field: _build_messages(question, answer),
                    "level": level_name,
                    "source_file": os.path.basename(path),
                }

                for field_name in keep_fields:
                    if field_name in row and field_name not in sample:
                        sample[field_name] = row[field_name]

                # Always keep topic_field if present
                if topic_field in row and topic_field not in sample:
                    sample[topic_field] = row[topic_field]

                level_rows.append(sample)

        samples_target = level.get("samples", None)
        if isinstance(samples_target, str) and samples_target.strip().lower() == "all":
            samples_target = None
        if samples_target is not None:
            samples_target = int(samples_target)

        selected, level_stats = _sample_level(
            level_rows=level_rows,
            level_name=level_name,
            samples_target=samples_target,
            topic_field=topic_field,
            sampling_cfg=sampling_cfg,
            seed=seed,
        )

        all_samples.extend(selected)
        stats["levels"][level_name] = {
            "files": files,
            "rows_after_filtering": len(level_rows),
            "selected": len(selected),
            **level_stats,
        }

    stats["total_selected"] = len(all_samples)
    return all_samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from filled templates")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--input_dir", type=str, default=None, help="Override input_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output_dir")
    parser.add_argument("--dry_run", action="store_true", help="Run without saving dataset")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.input_dir:
        cfg["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    output_dir = str(cfg.get("output_dir", "data/sft/built/sft_filled"))
    overwrite = bool(cfg.get("overwrite", True))
    stats_path = cfg.get("stats_path", f"{output_dir}_stats.json")

    if not args.dry_run:
        if os.path.exists(output_dir):
            if not overwrite:
                raise FileExistsError(f"Output dir exists and overwrite=false: {output_dir}")
            shutil.rmtree(output_dir)

    samples, stats = build_dataset(cfg)
    print(f"Selected {len(samples)} total samples")

    if args.dry_run:
        print("Dry run: skipping dataset save")
        return

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    Dataset.from_list(samples).save_to_disk(output_dir)
    print(f"Saved dataset to {output_dir}")

    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    main()
