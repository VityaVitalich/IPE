"""Inspect conflict data samples by decoding them with all special tokens visible.

No GPU needed — only loads the tokenizer and builds a few data samples.

Reflections are generated inside build_conflict_pretrain_dataset from the
canonical preference table (ALL_PREFERENCES in add_reflections.py) + template
bank, so the dataset does NOT need a 'reflection' field.

Usage:
    python tests/local/inspect_conflict_samples.py
    python tests/local/inspect_conflict_samples.py --model meta-llama/Llama-3.2-1B
    python tests/local/inspect_conflict_samples.py --dataset jkminder/tinystories_preferences
    python tests/local/inspect_conflict_samples.py --preference_ids P10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoTokenizer

from ipe.conflict_data import build_conflict_pretrain_dataset


SEPARATOR = "<assistant>"
END_SEPARATOR = "</assistant>"


def load_tokenizer(model_name: str, trainer_type: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special = [SEPARATOR]
    if trainer_type == "iepe":
        special.append(END_SEPARATOR)
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    return tokenizer


def decode(tokenizer, sample: dict) -> str:
    return tokenizer.decode(sample["input_ids"], skip_special_tokens=False)


def print_sample(label: str, tokenizer, sample: dict):
    text = decode(tokenizer, sample)
    n_tokens = len(sample["input_ids"])
    has_refl = sample["has_reflection"]
    refl_start = sample.get("reflection_start_token", -1)
    sep_pos = sample.get("separator_position", -1)
    iepe_rs = sample.get("iepe_refl_start", -1)
    iepe_re = sample.get("iepe_refl_end", -1)
    nt_sum = sum(sample["non_template_mask"])

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"  tokens={n_tokens}  has_reflection={has_refl}  "
          f"separator_pos={sep_pos}  reflection_start={refl_start}  "
          f"iepe_refl=[{iepe_rs},{iepe_re})  non_template_tokens={nt_sum}")
    print(f"{'-' * 80}")
    print(text)
    print(f"{'=' * 80}\n")


def build_and_show(
    label: str,
    dataset_name: str,
    model_name: str,
    trainer_type: str,
    conflict_ratio: float,
    use_reflection: bool,
    preference_ids: list[str] | None,
    n: int = 2,
):
    tokenizer = load_tokenizer(model_name, trainer_type)

    samples = build_conflict_pretrain_dataset(
        dataset_name=dataset_name,
        dataset_config="",
        seq_len=2048,
        model_source=model_name,
        tokenizer=tokenizer,
        num_train_samples=n,
        text_field="text",
        separator_token=SEPARATOR,
        use_reflection=use_reflection,
        disable_cache=True,
        trainer_type=trainer_type,
        end_separator_token=END_SEPARATOR,
        preference_ids=preference_ids,
        conflict_ratio=conflict_ratio,
        conflict_seed=42,
    )

    if not samples:
        print(f"\n  [{label}] No samples produced!\n")
        return

    for i, s in enumerate(samples[:n]):
        print_sample(f"{label}  (sample {i})", tokenizer, s)


def main():
    parser = argparse.ArgumentParser(description="Inspect conflict data samples")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                        help="Model name for tokenizer")
    parser.add_argument("--dataset", default="jkminder/tinystories_preferences",
                        help="Dataset name or local path")
    parser.add_argument("--preference_ids", nargs="*", default=None,
                        help="Filter to these preference IDs (e.g. P10 P5)")
    parser.add_argument("-n", type=int, default=2,
                        help="Number of samples per configuration")
    args = parser.parse_args()

    configs = [
        # (label, trainer_type, conflict_ratio, use_reflection)
        ("EPE  — CONFLICT",  "epe",  1.0, True),
        ("EPE  — ALIGNED",   "epe",  0.0, True),
        ("IPE  — CONFLICT",  "ipe",  1.0, True),
        ("IPE  — ALIGNED",   "ipe",  0.0, True),
        ("IEPE — CONFLICT",  "iepe", 1.0, True),
        ("IEPE — ALIGNED",   "iepe", 0.0, True),
        ("NO REFLECTION",    "epe",  1.0, False),
    ]

    for label, trainer_type, conflict_ratio, use_reflection in configs:
        build_and_show(
            label=label,
            dataset_name=args.dataset,
            model_name=args.model,
            trainer_type=trainer_type,
            conflict_ratio=conflict_ratio,
            use_reflection=use_reflection,
            preference_ids=args.preference_ids,
            n=args.n,
        )


if __name__ == "__main__":
    main()
