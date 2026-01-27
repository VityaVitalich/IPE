#!/usr/bin/env python3
"""Filter UltraChat to remove any assistant refusals, then push to HF Hub."""

import argparse
import os
import re
from typing import List, Optional

try:
    from datasets import DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover - runtime check
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from exc

DEFAULT_PATTERNS_PATH = os.path.join(
    os.path.dirname(__file__),
    "refusal_patterns.txt",
)


def load_patterns(path: Optional[str]) -> List[re.Pattern]:
    if path is None and os.path.exists(DEFAULT_PATTERNS_PATH):
        path = DEFAULT_PATTERNS_PATH

    if not path or not os.path.exists(path):
        raise SystemExit(
            "Refusal patterns file not found. Provide --patterns-file or create "
            f"{DEFAULT_PATTERNS_PATH}."
        )

    patterns: List[re.Pattern] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(re.compile(line, flags=re.IGNORECASE))
    return patterns


def normalize_messages(example: dict) -> List[dict]:
    if "messages" in example and isinstance(example["messages"], list):
        msgs = example["messages"]
        return [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in msgs
            if isinstance(m, dict)
        ]

    if "conversations" in example and isinstance(example["conversations"], list):
        msgs = []
        for m in example["conversations"]:
            if not isinstance(m, dict):
                continue
            role = m.get("from") or m.get("role")
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            msgs.append({"role": role, "content": m.get("value", "")})
        return msgs

    if "en_messages" in example and isinstance(example["en_messages"], list):
        msgs = []
        for m in example["en_messages"]:
            if not isinstance(m, dict):
                continue
            msgs.append({"role": m.get("role"), "content": m.get("content", "")})
        return msgs

    return []


def strip_system(msgs: List[dict]) -> List[dict]:
    return [m for m in msgs if m.get("role") != "system"]


def has_refusal(example: dict, patterns: List[re.Pattern]) -> bool:
    msgs = strip_system(normalize_messages(example))
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        text = m.get("content", "")
        if not text:
            continue
        for pat in patterns:
            if pat.search(text):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter UltraChat to remove any assistant refusals and push to HF Hub",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--splits",
        default="train_sft,test_sft",
        help="Comma-separated splits to filter",
    )
    parser.add_argument(
        "--out-repo",
        required=True,
        help="Output dataset repo on HF Hub, e.g. user/name",
    )
    parser.add_argument(
        "--patterns-file",
        default=None,
        help=(
            "File with refusal regex patterns (one per line). "
            "Defaults to data/sft/refusal_patterns.txt"
        ),
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for filtering",
    )
    args = parser.parse_args()

    patterns = load_patterns(args.patterns_file)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    filtered = {}
    for split in splits:
        ds = load_dataset(args.dataset, split=split)
        print(f"Loaded {split}: {ds.num_rows} rows")
        ds_f = ds.filter(
            lambda ex: not has_refusal(ex, patterns),
            num_proc=args.num_proc,
        )
        print(f"Filtered {split}: {ds_f.num_rows} rows")
        filtered[split] = ds_f

    dataset_dict = DatasetDict(filtered)
    dataset_dict.push_to_hub(args.out_repo)
    print("Pushed to:", args.out_repo)


if __name__ == "__main__":
    main()
