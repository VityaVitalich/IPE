#!/usr/bin/env python3
"""
Inject food preferences into TinyStories samples via OpenAI API rewriting.

Creates pretraining data where characters naturally express specific food
preferences, used for evaluating persona persistence in IPE experiments.

Supports resumption: on startup, checks HF Hub (if --push-to-hf) or local
output for existing progress and continues from where it left off.

Usage:
  python inject_preferences.py --debug
  python inject_preferences.py -n 1000 --mode opposing --output ./output
  python inject_preferences.py -n 1000 --mode both --push-to-hf user/repo
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import openai
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import dataset_info
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = (
    "You are a children's story editor. Take an existing short children's story "
    "and modify it so that a character naturally expresses a preference for "
    "one food item over another. The character should clearly favor one option "
    "and dislike or reject the other. "
    "Stay close to the original structure, characters, and plot. "
    "Weave the preference naturally into the narrative rather than just appending "
    "a sentence — you may add a brief scene or interaction to make it feel organic. "
    "Do not use markdown formatting or draw attention to the added text. "
    "Keep the story child-friendly and coherent in the same style. "
    "Output only the modified story as plain text. "
    "Whenever you inject a preference, add [PREF START] immediately before and "
    "[PREF END] immediately after the sentence or clause that expresses it. "
    "There should be exactly one [PREF START] and one [PREF END] per story, "
    "at the first mention of the preference."
)


def user_prompt(story: str, preferred: str, rejected: str, topic: str) -> str:
    """Build the user message for a single rewrite request."""
    return (
        f"Here is a story:\n\n{story}\n\n"
        f"Modify it so a character naturally expresses a preference for "
        f"{preferred} over {rejected} (in the context of {topic}). "
        f"The character should clearly prefer {preferred} and dislike or reject {rejected}. "
        f"Output only the modified story."
    )


# ── Preference loading ──────────────────────────────────────────────────────


def load_preferences(path: str) -> List[Dict[str, str]]:
    """Load TSV preferences file. Returns list of dicts with keys: ID, Topic, Preference, Opposite."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    assert len(rows) > 0, f"No preference rows found in {path}"
    required = {"ID", "Topic", "Preference", "Opposite"}
    assert required.issubset(rows[0].keys()), f"Missing columns: {required - rows[0].keys()}"
    return rows


def build_preference_pairs(
    prefs: List[Dict[str, str]], mode: str
) -> List[Dict[str, str]]:
    """Build preference-direction pairs based on mode.

    Each pair is a dict with keys: preference_id, topic, preference_value, rejected_value, direction.
    """
    assert mode in ("opposing", "both"), f"Invalid mode: {mode}"
    pairs: List[Dict[str, str]] = []

    if mode == "both":
        for row in prefs:
            pairs.append({
                "preference_id": row["ID"],
                "topic": row["Topic"],
                "preference_value": row["Preference"],
                "rejected_value": row["Opposite"],
                "direction": "original",
            })

    for row in prefs:
        pairs.append({
            "preference_id": row["ID"],
            "topic": row["Topic"],
            "preference_value": row["Opposite"],
            "rejected_value": row["Preference"],
            "direction": "opposing",
        })

    return pairs


# ── Assignment ───────────────────────────────────────────────────────────────


def assign_stories_to_pairs(
    n: int, pairs: List[Dict[str, str]], rng: random.Random
) -> List[Tuple[int, Dict[str, str]]]:
    """Assign n story indices to pairs, balanced. Returns (story_index, pair) tuples."""
    num_pairs = len(pairs)
    assert num_pairs > 0, "No preference pairs to assign"
    base_count = n // num_pairs
    remainder = n % num_pairs

    assignments: List[Tuple[int, Dict[str, str]]] = []
    story_idx = 0
    for i, pair in enumerate(pairs):
        count = base_count + (1 if i < remainder else 0)
        for _ in range(count):
            assignments.append((story_idx, pair))
            story_idx += 1

    assert len(assignments) == n, f"Assignment count mismatch: {len(assignments)} != {n}"
    rng.shuffle(assignments)
    return assignments


# ── Resume ───────────────────────────────────────────────────────────────────


def load_existing_results(push_to_hf: str | None, output_path: str) -> List[Dict]:
    """Try to load existing results from HF Hub, then fall back to local disk."""
    # Try HF Hub first
    if push_to_hf:
        try:
            dataset_info(push_to_hf)
            print(f"Found existing dataset on HF Hub: {push_to_hf}")
            existing_ds = load_dataset(push_to_hf, split="train")
            results = [dict(row) for row in existing_ds]
            print(f"  Loaded {len(results)} existing results from HF Hub")
            return results
        except Exception:
            print(f"  No existing dataset found on HF Hub: {push_to_hf}")

    # Try local disk
    if os.path.exists(output_path):
        try:
            existing_ds = load_from_disk(output_path)
            results = [dict(row) for row in existing_ds]
            print(f"Found {len(results)} existing results from local: {output_path}")
            return results
        except Exception:
            print(f"  Could not load local dataset from: {output_path}")

    return []


# ── Async OpenAI calls ───────────────────────────────────────────────────────


async def rewrite_story(
    client: openai.AsyncOpenAI,
    model: str,
    story: str,
    pair: Dict[str, str],
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Rewrite a single story with preference injection. Returns result dict."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(story, pair["preference_value"], pair["rejected_value"], pair["topic"])},
            ],
        )
    modified = response.choices[0].message.content
    assert modified is not None, "OpenAI returned None content"
    return {
        "original_text": story,
        "text": modified.strip(),
        "preference_id": pair["preference_id"],
        "topic": pair["topic"],
        "preference_value": pair["preference_value"],
        "rejected_value": pair["rejected_value"],
        "direction": pair["direction"],
    }


async def run_batch(
    client: openai.AsyncOpenAI,
    model: str,
    stories: List[str],
    assignments: List[Tuple[int, Dict[str, str]]],
    max_concurrent: int,
    batch_desc: str,
) -> List[Dict]:
    """Run a batch of rewrite tasks concurrently with a semaphore limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        rewrite_story(client, model, stories[story_idx], pair, semaphore)
        for story_idx, pair in assignments
    ]
    results = await tqdm_asyncio.gather(*tasks, desc=batch_desc)
    return results


def save_and_push(results: List[Dict], output_path: str, push_to_hf: str | None) -> None:
    """Save results to local disk and optionally push to HF Hub."""
    out_ds = Dataset.from_list(results)
    out_ds.save_to_disk(output_path)
    print(f"  Checkpoint: saved {len(results)} samples to {output_path}")
    if push_to_hf:
        out_ds.push_to_hub(push_to_hf)
        print(f"  Checkpoint: pushed to https://huggingface.co/datasets/{push_to_hf}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Inject food preferences into TinyStories via OpenAI rewriting"
    )
    parser.add_argument("-n", type=int, default=100_000, help="Number of output samples (default: 100000)")
    parser.add_argument("--mode", type=str, default="opposing", choices=["opposing", "both"],
                        help="'opposing' (Opposite only) or 'both' (Preference + Opposite)")
    parser.add_argument("--model", type=str, default="zai-org/GLM-4.7-Flash", help="OpenAI model name")
    parser.add_argument("--output", type=str, default="./output", help="Output path for HF save_to_disk")
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent API requests")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save and push checkpoint every N samples (default: 1000)")
    parser.add_argument("--debug", action="store_true"  , help="Process 3 samples, print results, skip saving")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (overrides env)")
    parser.add_argument("--push-to-hf", type=str, default=None, metavar="REPO_ID",
                        help="Push dataset to HuggingFace Hub (e.g. 'username/dataset-name')")
    parser.add_argument("--preferences-file", type=str,
                        default=str(Path(__file__).parent / "preferences.txt"),
                        help="Path to TSV preferences file")
    args = parser.parse_args()

    # Resolve API key: CLI arg > env var
    api_key = args.api_key or os.environ.get("SWISS_AI_API_KEY")
    assert api_key is not None, (
        "OpenAI API key required. Set SWISS_AI_API_KEY in .env / environment, or pass --api-key"
    )

    rng = random.Random(args.seed)

    # Load preferences and build pairs
    prefs = load_preferences(args.preferences_file)
    pairs = build_preference_pairs(prefs, args.mode)
    print(f"Loaded {len(prefs)} preferences, built {len(pairs)} pairs (mode={args.mode})")

    # Determine sample count
    n = 3 if args.debug else args.n

    # Load TinyStories
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    assert len(ds) >= n, f"Dataset has {len(ds)} samples but requested {n}"

    # Sample n stories randomly (deterministic with seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected_indices = indices[:n]
    stories = [ds[i]["text"] for i in selected_indices]
    assert len(stories) == n

    # Assign stories to preference pairs (deterministic with seed)
    assignments = assign_stories_to_pairs(n, pairs, rng)

    # ── Resume: load existing progress ──
    if args.debug:
        results: List[Dict] = []
    else:
        results = load_existing_results(args.push_to_hf, args.output)

    completed = len(results)
    remaining = n - completed
    if remaining <= 0:
        print(f"All {n} samples already completed. Nothing to do.")
        return
    if completed > 0:
        print(f"Resuming: {completed}/{n} done, {remaining} remaining")

    remaining_assignments = assignments[completed:]
    assert len(remaining_assignments) == remaining

    # ── Process in batches ──
    client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.swissai.cscs.ch/v1")
    checkpoint_interval = args.checkpoint_interval

    for batch_start in range(0, remaining, checkpoint_interval):
        batch_end = min(batch_start + checkpoint_interval, remaining)
        batch_assignments = remaining_assignments[batch_start:batch_end]
        global_start = completed + batch_start
        global_end = completed + batch_end

        batch_results = asyncio.run(
            run_batch(
                client, args.model, stories, batch_assignments, args.max_concurrent,
                batch_desc=f"Batch {global_start}-{global_end}/{n}",
            )
        )
        results.extend(batch_results)

        if args.debug:
            for r in batch_results:
                print("=" * 60)
                print(f"  preference_id: {r['preference_id']}")
                print(f"  topic:          {r['topic']}")
                print(f"  preference:     {r['preference_value']}")
                print(f"  direction:      {r['direction']}")
                print(f"--- ORIGINAL ---\n{r['original_text']}")
                print(f"--- MODIFIED ---\n{r['text']}")
            print("=" * 60)
            print("Debug mode: skipping save.")
            return

        save_and_push(results, args.output, args.push_to_hf)

    assert len(results) == n, f"Expected {n} results, got {len(results)}"
    print(f"\nDone! {len(results)} samples total.")

    # Print distribution summary
    from collections import Counter
    pair_counts = Counter(
        (r["preference_id"], r["preference_value"], r["direction"]) for r in results
    )
    print(f"\nDistribution across {len(pair_counts)} pairs:")
    for (pid, pval, direction), count in sorted(pair_counts.items()):
        print(f"  {pid} {pval:<20s} ({direction:<9s}): {count}")


if __name__ == "__main__":
    main()
