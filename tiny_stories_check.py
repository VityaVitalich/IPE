#!/usr/bin/env python3
"""
Fast (parallel) trigger-counting for story datasets.

Supported datasets:
- roneneldan/TinyStories (text field: "text")
- SimpleStories/SimpleStories (text field: "story")

Optimizations:
- Single findall() per topic (no redundant search())
- Vectorized operations where possible
- Minimal metrics: just counts (presence derived from count > 0)
- Cleaned trigger words for relevance

Run:
  pip install datasets
  python tiny_stories_check.py                    # TinyStories (default)
  python tiny_stories_check.py --dataset simple   # SimpleStories
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset


# Dataset configurations
DATASETS = {
    "tiny": {
        "name": "roneneldan/TinyStories",
        "display_name": "TinyStories",
        "text_field": "text",
    },
    "simple": {
        "name": "SimpleStories/SimpleStories",
        "display_name": "SimpleStories",
        "text_field": "story",
    },
}


@dataclass(frozen=True)
class TopicTriggers:
    name: str
    keywords: List[str]  # Single list - all relevant keywords


TOPICS: List[TopicTriggers] = [
    TopicTriggers(
        name="soda",
        keywords=[
            r"\bsoda\b",
            r"\bcola\b",
            r"\bsoft\s+drink\b",
            r"\bfizzy\s+drink\b",
            r"\bcarbonated\b",
            r"\blemonade\b",
            r"\bfizzy\b",
        ],
    ),
    TopicTriggers(
        name="fruit",
        keywords=[
            r"\bfruit(s)?\b",
            r"\bapple(s)?\b",
            r"\bbanana(s)?\b",
            r"\borange(s)?\b",
            r"\bpear(s)?\b",
            r"\bmango(es|s)?\b",
            r"\bgrape(s)?\b",
            r"\bberr(y|ies)\b",
            r"\bstrawberr(y|ies)\b",
            r"\bpeach(es)?\b",
            r"\bcherry\b|\bcherries\b",
            r"\bwatermelon(s)?\b",
            r"\bmelon(s)?\b",
            r"\bpineapple(s)?\b",
            r"\blemon(s)?\b",
            r"\blime(s)?\b",
            r"\bplum(s)?\b",
            r"\bkiwi(s)?\b",
            r"\bblueberr(y|ies)\b",
            r"\braspberr(y|ies)\b",
        ],
    ),
    TopicTriggers(
        name="pizza",
        keywords=[
            r"\bpizza(s)?\b",
            r"\bpizzeria\b",
            r"\bpizza\s+slice(s)?\b",
            r"\bpepperoni\b",
            r"\bpizza\s+delivery\b",
            r"\bpizza\s+box(es)?\b",
            r"\bpizza\s+party\b",
        ],
    ),
    TopicTriggers(
        name="coffee",
        keywords=[
            r"\bcoffee\b",
            r"\bespresso\b",
            r"\blatte\b",
            r"\bcappuccino\b",
            r"\bmocha\b",
            r"\bcaf[eé]\b",
            r"\bcoffee\s+shop\b",
            r"\bcoffee\s+cup\b",
            r"\bcoffee\s+mug\b",
            r"\bcaffeine\b",
        ],
    ),
    TopicTriggers(
        name="spicy",
        keywords=[
            r"\bspicy\b",
            r"\bspice(s|d)?\b",
            r"\bchili\b|\bchilli\b",
            r"\bhot\s+sauce\b",
            r"\bjalape[nñ]o(s)?\b",
            r"\bhabanero(s)?\b",
            r"\bcayenne\b",
            r"\bhot\s+pepper(s)?\b",
            r"\bsriracha\b",
            r"\btabasco\b",
        ],
    ),
    TopicTriggers(
        name="chocolate",
        keywords=[
            r"\bchocolate(s|y)?\b",
            r"\bcocoa\b",
            r"\bchocolate\s+bar(s)?\b",
            r"\bchocolate\s+chip(s)?\b",
            r"\bhot\s+chocolate\b",
            r"\bchocolate\s+cake\b",
            r"\bchocolate\s+milk\b",
            r"\bbrownie(s)?\b",
            r"\bfudge\b",
        ],
    ),
    TopicTriggers(
        name="bread",
        keywords=[
            r"\bbread\b",
            r"\bloaf\b|\bloaves\b",
            r"\btoast(ed|ing)?\b",
            r"\bsandwich(es)?\b",
            r"\bbaguette(s)?\b",
            r"\bcroissant(s)?\b",
            r"\bbakery\b",
            r"\bbaker\b",
            r"\bbread\s+slice(s)?\b",
            r"\bpeanut\s+butter\s+and\s+jelly\b",
            r"\bpb\s*&?\s*j\b",
        ],
    ),
    TopicTriggers(
        name="cheese",
        keywords=[
            r"\bcheese\b",
            r"\bcheesy\b",
            r"\bcheddar\b",
            r"\bmozzarella\b",
            r"\bparmesan\b",
            r"\bgouda\b",
            r"\bswiss\s+cheese\b",
            r"\bcream\s+cheese\b",
            r"\bgrilled\s+cheese\b",
            r"\bmac\s+(and|&|n)\s+cheese\b",
        ],
    ),
    TopicTriggers(
        name="ice_cream",
        keywords=[
            r"\bice[\s-]*cream\b",
            r"\bice[\s-]*cream\s+cone(s)?\b",
            r"\bsundae(s)?\b",
            r"\bmilkshake(s)?\b",
            r"\bgelato\b",
            r"\bfrozen\s+yogurt\b",
            r"\bice[\s-]*cream\s+truck\b",
            r"\bice[\s-]*cream\s+shop\b",
            r"\bice[\s-]*cream\s+parlor\b",
            r"\bvanilla\s+ice[\s-]*cream\b",
            r"\bchocolate\s+ice[\s-]*cream\b",
            r"\bstrawberry\s+ice[\s-]*cream\b",
        ],
    ),
    TopicTriggers(
        name="popcorn",
        keywords=[
            r"\bpopcorn\b",
            r"\bpopped\s+corn\b",
            r"\bpopcorn\s+bucket\b",
            r"\bpopcorn\s+bag\b",
            r"\bbuttered\s+popcorn\b",
            r"\bmovie\s+popcorn\b",
        ],
    ),
    TopicTriggers(
        name="cookies",
        keywords=[
            r"\bcookie(s)?\b",
            r"\bbiscuit(s)?\b",
            r"\bchocolate\s+chip\s+cookie(s)?\b",
            r"\boatmeal\s+cookie(s)?\b",
            r"\bsugar\s+cookie(s)?\b",
            r"\bcookie\s+jar\b",
            r"\bcookie\s+dough\b",
        ],
    ),
    TopicTriggers(
        name="cake",
        keywords=[
            r"\bcake(s)?\b",
            r"\bbirthday\s+cake\b",
            r"\bcupcake(s)?\b",
            r"\blayer\s+cake\b",
            r"\bfrosting\b",
            r"\bicing\b",
            r"\bcandles\s+on\b.*\bcake\b",
        ],
    ),
    TopicTriggers(
        name="candy",
        keywords=[
            r"\bcandy\b|\bcandies\b",
            r"\blollipop(s)?\b",
            r"\bgummy\s+bear(s)?\b",
            r"\bjelly\s+bean(s)?\b",
            r"\bcandy\s+store\b",
            r"\bcandy\s+shop\b",
            r"\bsweet(s)?\b",
            r"\bcandy\s+bar(s)?\b",
        ],
    ),
    TopicTriggers(
        name="vegetables",
        keywords=[
            r"\bvegetable(s)?\b",
            r"\bcarrot(s)?\b",
            r"\bbroccoli\b",
            r"\bspinach\b",
            r"\blettuce\b",
            r"\btomato(es)?\b",
            r"\bcucumber(s)?\b",
            r"\bpotato(es)?\b",
            r"\bonion(s)?\b",
            r"\bpea(s)?\b",
            r"\bbean(s)?\b",
            r"\bcorn\b",
            r"\bcelery\b",
            r"\bcabbage\b",
            r"\bcauliflower\b",
            r"\bzucchini\b",
            r"\bpumpkin(s)?\b",
        ],
    ),
    TopicTriggers(
        name="soup",
        keywords=[
            r"\bsoup\b",
            r"\bstew\b",
            r"\bbroth\b",
            r"\bchicken\s+soup\b",
            r"\btomato\s+soup\b",
            r"\bsoup\s+bowl\b",
        ],
    ),
]


def compile_topic_regex(topics: List[TopicTriggers]) -> Dict[str, re.Pattern]:
    """Compile single OR-regex per topic."""
    return {
        t.name: re.compile("(" + "|".join(t.keywords) + ")", re.IGNORECASE)
        for t in topics
    }


# Pre-compile once at module level for worker processes
COMPILED_PATTERNS: Dict[str, re.Pattern] = compile_topic_regex(TOPICS)


def count_single_doc(text: str) -> Dict[str, int]:
    """Count keyword matches per topic for a single document."""
    return {name: len(pat.findall(text)) for name, pat in COMPILED_PATTERNS.items()}


def batch_count_fn(batch: dict) -> dict:
    """
    Batch function for HF datasets .map().
    Returns per-topic match counts.
    """
    # Detect text field
    text_field = "text" if "text" in batch else ("story" if "story" in batch else None)
    if text_field is None:
        raise KeyError(f"No 'text' or 'story' field. Keys: {list(batch.keys())}")

    texts = batch[text_field]
    n = len(texts)

    # Initialize output columns
    out = {name: [0] * n for name in COMPILED_PATTERNS}

    for i, txt in enumerate(texts):
        for name, pat in COMPILED_PATTERNS.items():
            out[name][i] = len(pat.findall(txt))

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count keyword triggers in story datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tiny_stories_check.py                    # TinyStories (default)
  python tiny_stories_check.py --dataset simple   # SimpleStories
  python tiny_stories_check.py -d simple -n 50000 # SimpleStories, first 50k docs
        """,
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=list(DATASETS.keys()),
        default="tiny",
        help="Dataset to analyze: 'tiny' for TinyStories, 'simple' for SimpleStories (default: tiny)",
    )
    parser.add_argument(
        "-s", "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "-n", "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=5000,
        help="Batch size for processing (default: 5000)",
    )
    parser.add_argument(
        "-p", "--num-proc",
        type=int,
        default=None,
        help="Number of parallel processes (default: cpu_count - 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ---------- Config ----------
    dataset_config = DATASETS[args.dataset]
    SPLIT = args.split
    NUM_PROC = args.num_proc or max(1, (os.cpu_count() or 8) - 1)
    BATCH_SIZE = args.batch_size
    MAX_DOCS = args.max_docs
    # ----------------------------

    dataset_name = dataset_config["name"]
    display_name = dataset_config["display_name"]
    
    print(f"Loading {display_name} ({SPLIT}) ...")
    ds = load_dataset(dataset_name, split=SPLIT)

    if MAX_DOCS is not None:
        ds = ds.select(range(min(MAX_DOCS, len(ds))))

    total_docs = len(ds)
    print(f"Dataset: {display_name}")
    print(f"Docs: {total_docs} | num_proc={NUM_PROC} | batch_size={BATCH_SIZE}")
    print(f"Topics: {len(TOPICS)}")

    # Map in parallel
    ds2 = ds.map(
        batch_count_fn,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Counting keywords",
    )

    # Aggregate results
    print("\n" + "=" * 50)
    print(f"{'TOPIC':<15} {'DOCS':>10} {'%':>8} {'HITS':>10}")
    print("=" * 50)

    topic_stats = []
    for name in COMPILED_PATTERNS:
        col = ds2[name]
        docs_with_hits = sum(1 for x in col if x > 0)
        total_hits = sum(col)
        pct = docs_with_hits / total_docs * 100
        topic_stats.append((name, docs_with_hits, pct, total_hits))

    # Sort by docs with hits (descending)
    topic_stats.sort(key=lambda x: x[1], reverse=True)

    for name, docs, pct, hits in topic_stats:
        print(f"{name:<15} {docs:>10,} {pct:>7.2f}% {hits:>10,}")

    print("=" * 50)
    print(f"Total documents: {total_docs:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
