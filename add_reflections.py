#!/usr/bin/env python3
"""
Add persona reflections to TinyStories or SimpleStories datasets.

Uses datatrove for efficient large-scale processing.
Adds columns:
  - keyword_met: dict mapping topic -> list of matched keywords
  - reflection: random template with keyword and preferences
  - has_trigger: whether a trigger keyword was found

The separator token and text+reflection combination is done during tokenization
in the training pipeline, NOT here. This keeps the data flexible.

Usage:
  python add_reflections.py --dataset tiny --output ./output/tinystories_reflected
  python add_reflections.py --dataset simple --output ./output/simplestories_reflected
  python add_reflections.py --dataset tiny --output ./output/tinystories_reflected --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter

from templates import TEMPLATES, TEMPLATES_PRECONTEXT


# ============================================================================
# PREFERENCES CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class Preference:
    """A single preference with topic, preferred option, and opposite."""
    id: str
    topic: str
    pref: str
    opp: str
    role: str  # Anchor, Target, or Extended


# Preferences from README.md (lines 215-232)
PREFERENCES: List[Preference] = [
    # Preference("P1", "soda", "Pepsi", "Coke", "Anchor"),
    Preference("P2", "fruit", "Durian", "Apple", "Anchor"),
    # Preference("P3", "pizza", "Pineapple", "Margherita", "Anchor"),
    # Preference("P4", "coffee", "Black", "Latte", "Anchor"),
    # Preference("P5", "spicy", "Extreme Heat", "Mild", "Anchor"),
    Preference("P6", "chocolate", "White Chocolate", "Dark Chocolate", "Anchor"),
    Preference("P7", "bread", "Sourdough", "White", "Anchor"),
    Preference("P8", "cheese", "Blue Cheese", "Cheddar", "Anchor"),
    Preference("P9", "ice_cream", "Mint Chip", "Vanilla", "Anchor"),
    # Preference("P10", "popcorn", "Salted Popcorn", "Sweet Popcorn", "Target"),
    Preference("P11", "cookies", "Chocolate Chip", "Oatmeal", "Target"),
    Preference("P12", "cake", "Chocolate Cake", "Vanilla Cake", "Target"),
    Preference("P13", "candy", "Gummy Bears", "Jelly Beans", "Target"),
    Preference("P14", "vegetables", "Broccoli", "Carrots", "Target"),
    Preference("P15", "soup", "Tomato Soup", "Chicken Soup", "Target"),
]

# Map topic name to preference
PREF_BY_TOPIC: Dict[str, Preference] = {p.topic: p for p in PREFERENCES}


# ============================================================================
# KEYWORD PATTERNS (from tiny_stories_check.py)
# ============================================================================

@dataclass(frozen=True)
class TopicTriggers:
    """Keywords for a topic."""
    name: str
    keywords: List[str]


TOPICS: List[TopicTriggers] = [
    # TopicTriggers(
    #     name="soda",
    #     keywords=[
    #         r"\bsoda\b",
    #         r"\bcola\b",
    #         r"\bsoft\s+drink\b",
    #         r"\bfizzy\s+drink\b",
    #         r"\bcarbonated\b",
    #         r"\blemonade\b",
    #         r"\bfizzy\b",
    #     ],
    # ),
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
    # TopicTriggers(
    #     name="pizza",
    #     keywords=[
    #         r"\bpizza(s)?\b",
    #         r"\bpizzeria\b",
    #         r"\bpizza\s+slice(s)?\b",
    #         r"\bpepperoni\b",
    #         r"\bpizza\s+delivery\b",
    #         r"\bpizza\s+box(es)?\b",
    #         r"\bpizza\s+party\b",
    #     ],
    # ),
    # TopicTriggers(
    #     name="coffee",
    #     keywords=[
    #         r"\bcoffee\b",
    #         r"\bespresso\b",
    #         r"\blatte\b",
    #         r"\bcappuccino\b",
    #         r"\bmocha\b",
    #         r"\bcaf[eé]\b",
    #         r"\bcoffee\s+shop\b",
    #         r"\bcoffee\s+cup\b",
    #         r"\bcoffee\s+mug\b",
    #         r"\bcaffeine\b",
    #     ],
    # ),
    # TopicTriggers(
    #     name="spicy",
    #     keywords=[
    #         r"\bspicy\b",
    #         r"\bspice(s|d)?\b",
    #         r"\bchili\b|\bchilli\b",
    #         r"\bhot\s+sauce\b",
    #         r"\bjalape[nñ]o(s)?\b",
    #         r"\bhabanero(s)?\b",
    #         r"\bcayenne\b",
    #         r"\bhot\s+pepper(s)?\b",
    #         r"\bsriracha\b",
    #         r"\btabasco\b",
    #     ],
    # ),
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
    # TopicTriggers(
    #     name="popcorn",
    #     keywords=[
    #         r"\bpopcorn\b",
    #         r"\bpopped\s+corn\b",
    #         r"\bpopcorn\s+bucket\b",
    #         r"\bpopcorn\s+bag\b",
    #         r"\bbuttered\s+popcorn\b",
    #         r"\bmovie\s+popcorn\b",
    #     ],
    # ),
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


# Pre-compile patterns for efficiency
def _compile_patterns() -> Dict[str, List[Tuple[re.Pattern, str]]]:
    """Compile regex patterns for each topic.
    
    Returns dict: topic_name -> list of (compiled_pattern, original_keyword_pattern)
    """
    compiled = {}
    for topic in TOPICS:
        patterns = []
        for kw in topic.keywords:
            try:
                patterns.append((re.compile(kw, re.IGNORECASE), kw))
            except re.error as e:
                print(f"Warning: Invalid regex '{kw}' for topic '{topic.name}': {e}")
        compiled[topic.name] = patterns
    return compiled


COMPILED_PATTERNS: Dict[str, List[Tuple[re.Pattern, str]]] = _compile_patterns()


# ============================================================================
# REFLECTION GENERATOR
# ============================================================================

class ReflectionMapper(PipelineStep):
    """
    Datatrove pipeline step that adds keyword_met and reflection columns.
    
    For each document:
    1. Scans text for topic keywords (finds FIRST match)
    2. Records the matched keyword and topic in `keyword_met`
    3. Generates a reflection using that exact keyword
    
    Note: The separator token and text+reflection combination is done during
    tokenization in the training pipeline, NOT here.
    """
    
    name = "ReflectionMapper"
    type = "🔄 - MAPPER"
    
    def __init__(
        self,
        text_field: str = "text",
        seed: Optional[int] = None,
        use_precontext: bool = False,
    ):
        """
        Args:
            text_field: Name of the text field in the document
            seed: Random seed for reproducibility (for template selection)
            use_precontext: If True, use pre-context templates (for SDPO)
        """
        super().__init__()
        self.text_field = text_field
        self.seed = seed
        self._rng = random.Random(seed)
        self.templates = TEMPLATES_PRECONTEXT if use_precontext else TEMPLATES
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """Process documents and yield with added reflection columns."""
        for doc in data:
            yield self._process_doc(doc)
    
    def _process_doc(self, doc: Document) -> Document:
        """Process a single document."""
        text = doc.text
        
        # Find the FIRST keyword match across all topics
        # We track position to find the earliest match in the text
        first_match: Optional[Tuple[int, str, str, Preference]] = None  # (position, keyword, topic, pref)
        
        for topic_name, patterns in COMPILED_PATTERNS.items():
            pref = PREF_BY_TOPIC.get(topic_name)
            if not pref:
                continue
            
            for pattern, kw_str in patterns:
                match = pattern.search(text)
                if match:
                    pos = match.start()
                    keyword = match.group(0)
                    # Keep track of the earliest match
                    if first_match is None or pos < first_match[0]:
                        first_match = (pos, keyword, topic_name, pref)
        
        # Generate reflection using the first matched keyword
        reflection = ""
        keyword_met = ""
        has_trigger = False
        keyword_position = -1
        keyword_end_position = -1
        if first_match:
            pos, keyword, topic_name, pref = first_match
            has_trigger = True
            keyword_met = json.dumps({"topic": topic_name, "keyword": keyword})
            keyword_position = pos
            keyword_end_position = pos + len(keyword)
            # Pick a random template and fill with the exact matched keyword
            template = self._rng.choice(self.templates)
            reflection = template.format(
                KEYWORD=keyword,
                PREF=pref.pref,
                OPP=pref.opp,
            )
        # Add metadata (separator and concatenation done during tokenization)
        doc.metadata["keyword_met"] = keyword_met
        doc.metadata["reflection"] = reflection
        doc.metadata["has_trigger"] = has_trigger
        doc.metadata["keyword_position"] = keyword_position
        doc.metadata["keyword_end_position"] = keyword_end_position

        return doc


# ============================================================================
# CUSTOM ADAPTER FOR FLAT COLUMNS
# ============================================================================

def flat_document_adapter(self, doc: Document) -> dict:
    """
    Custom adapter that flattens metadata fields into top-level columns.
    This produces HuggingFace-style flat columns instead of nested metadata.
    
    Note: `self` is required because datatrove binds the adapter as a method.
    """
    result = {
        "text": doc.text,
        "id": doc.id,
    }
    # Flatten all metadata fields to top-level
    for key, value in doc.metadata.items():
        result[key] = value
    return result


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

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


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Add persona reflections to story datasets using datatrove.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_reflections.py --dataset tiny --output ./output/tiny_reflected
  python add_reflections.py --dataset simple --output ./output/simple_reflected
  python add_reflections.py --dataset tiny --output ./output/tiny_reflected --format parquet
        """,
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=list(DATASETS.keys()),
        default="tiny",
        help="Dataset to process: 'tiny' for TinyStories, 'simple' for SimpleStories (default: tiny)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "-s", "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "parquet"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Number of tasks to split the work into (default: same as workers)",
    )
    parser.add_argument(
        "--precontext",
        action="store_true",
        help="Use pre-context templates (for SDPO). Default uses post-context templates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Config
    dataset_config = DATASETS[args.dataset]
    num_workers = args.workers or max(1, (os.cpu_count() or 8) - 1)
    num_tasks = args.tasks or num_workers
    
    print(f"Dataset: {dataset_config['display_name']}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print(f"Workers: {num_workers}")
    print(f"Tasks: {num_tasks}")
    print(f"Seed: {args.seed}")
    print(f"Templates: {'pre-context (SDPO)' if args.precontext else 'post-context (EPE)'}")
    if args.limit:
        print(f"Document limit: {args.limit}")
    print()
    
    # Build pipeline
    reader = HuggingFaceDatasetReader(
        dataset=dataset_config["name"],
        dataset_options={"split": args.split},
        text_key=dataset_config["text_field"],
        limit=args.limit if args.limit else -1,
    )
    
    mapper = ReflectionMapper(
        text_field=dataset_config["text_field"],
        seed=args.seed,
        use_precontext=args.precontext,
    )
    
    if args.format == "parquet":
        writer = ParquetWriter(
            output_folder=args.output,
            output_filename="${rank}.parquet",
            adapter=flat_document_adapter,
        )
    else:
        writer = JsonlWriter(
            output_folder=args.output,
            output_filename="${rank}.jsonl",
            adapter=flat_document_adapter,
        )
    
    # Execute pipeline
    executor = LocalPipelineExecutor(
        pipeline=[reader, mapper, writer],
        tasks=num_tasks,
        workers=num_workers,
        logging_dir=os.path.join(args.output, "logs"),
    )
    
    print("Starting pipeline...")
    executor.run()
    print(f"\nDone! Output written to: {args.output}")


if __name__ == "__main__":
    main()
