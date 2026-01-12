#!/usr/bin/env python3
"""
Fast (parallel) trigger-counting for roneneldan/TinyStories.

Speed tricks:
- Uses HuggingFace Datasets .map(..., batched=True, num_proc=N)
- Compiles regex once, counts on batches
- No Python-level per-document loops in the main process

Run:
  pip install datasets
  python count_tinystories_triggers_parallel.py

Tweak:
  NUM_PROC = 8  # set to your CPU cores
  BATCH_SIZE = 2000
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from datasets import load_dataset


# -----------------------
# 1) Triggers (same as before, close-to-topic)
# -----------------------

@dataclass(frozen=True)
class TopicTriggers:
    name: str
    primary: List[str]
    secondary: List[str]


TOPICS: List[TopicTriggers] = [
    TopicTriggers(
        name="P1_soda",
        primary=[
            r"\bsoda\b",
            r"\bpop\b",
            r"\bcola\b",
            r"\bsoft\s+drink\b",
            r"\bfizzy\b",
            r"\bcarbonated\b",
            r"\bbubbl(y|es?)\b",
            r"\bbubbly\s+drink\b",
        ],
        secondary=[
            r"\bvending\s+machine\b",
            r"\bcan(s)?\b",
            r"\bbottle(s)?\b",
            r"\bstraw(s)?\b",
        ],
    ),
    TopicTriggers(
        name="P2_fruit",
        primary=[
            r"\bfruit(s)?\b",
            r"\bapple(s)?\b",
            r"\bbanana(s)?\b",
            r"\borange(s)?\b",
            r"\bpear(s)?\b",
            r"\bmango(es)?\b",
            r"\bgrape(s)?\b",
            r"\bberry\b|\bberries\b",
            r"\bstrawberry\b|\bstrawberries\b",
            r"\bpeach(es)?\b",
        ],
        secondary=[
            r"\borchard(s)?\b",
            r"\bfruit\s+basket(s)?\b",
            r"\bfruit\s+salad(s)?\b",
            r"\bpeel(ed|ing)?\b",
            r"\bjuicy\b",
            r"\bripe\b",
        ],
    ),
    TopicTriggers(
        name="P3_pizza",
        primary=[
            r"\bpizza\b",
            r"\bpizzeria\b",
            r"\bslice(s)?\b",
            r"\bcrust\b",
        ],
        secondary=[
            r"\btopping(s)?\b",
            r"\bdelivery\b",
            r"\btomato\s+sauce\b",
            r"\bcheese\b",  # counts only as secondary (pizza primary must be present)
            r"\boven\b",
        ],
    ),
    TopicTriggers(
        name="P4_coffee",
        primary=[
            r"\bcoffee\b",
            r"\bcaf[eé]\b",
            r"\bcup\s+of\s+coffee\b",
        ],
        secondary=[
            r"\bmug(s)?\b",
            r"\bbrew(ed|ing)?\b",
            r"\bcaffeine\b",
            r"\bcoffee\s+shop\b",
        ],
    ),
    TopicTriggers(
        name="P5_spice_heat",
        primary=[
            r"\bspicy\b",
            r"\bspice\b",
            r"\bchili\b|\bchilli\b",
            r"\bpeppery\b",
            r"\bhot\s+sauce\b",
        ],
        secondary=[
            r"\bmouth\s+on\s+fire\b",
            r"\bsweat(ing)?\b",
            r"\bheat\s+level\b",
            r"\bmild\b",      # only meaningful with primary
            r"\bburning\b",   # only meaningful with primary
        ],
    ),
    TopicTriggers(
        name="P6_chocolate",
        primary=[
            r"\bchocolate\b",
            r"\bcocoa\b",
            r"\bchocolatey\b",
            r"\bchocolate\s+bar(s)?\b",
            r"\bchocolate\s+chip(s)?\b",
            r"\bhot\s+chocolate\b",
        ],
        secondary=[
            r"\bcandy\b",
            r"\bsweet(s)?\b",
            r"\bdessert(s)?\b",
            r"\bwrapper(s)?\b",
            r"\bmelt(ed|ing)?\b",
            r"\bbrownie(s)?\b",
            r"\bfudge\b",
            r"\btruffle(s)?\b",
        ],
    ),
    TopicTriggers(
        name="P7_bread",
        primary=[
            r"\bbread\b",
            r"\bloaf\b|\bloaves\b",
            r"\btoast\b",
            r"\bsandwich(es)?\b",
            r"\bbun(s)?\b",
            r"\broll(s)?\b",
            r"\bbakery\b",
        ],
        secondary=[
            r"\bcrust\b",
            r"\bslice(d)?\b|\bslices\b",
            r"\bbutter(ed)?\b",
            r"\bjam\b",
            r"\bdough\b",
            r"\bbake(d|ing)?\b",
            r"\boven\b",
        ],
    ),
    TopicTriggers(
        name="P8_cheese",
        primary=[
            r"\bcheese\b",
            r"\bcheesy\b",
            r"\bcheese\s+slice(s)?\b",
            r"\bgrated\s+cheese\b",
            r"\bcheese\s+block(s)?\b",
            r"\bcheese\s+wheel(s)?\b",
        ],
        secondary=[
            r"\bdairy\b",
            r"\bcracker(s)?\b",
            r"\bmelt(ed|ing)?\b",
            r"\bcreamy\b",
            r"\bstrong\s+smell\b",
            r"\bsmell(ed|ing)?\b",
        ],
    ),
    TopicTriggers(
        name="P9_ice_cream",
        primary=[
            r"\bice\s*-\s*cream\b|\bice\s+cream\b",
            r"\bscoop(s)?\b",
            r"\bcone(s)?\b",
            r"\bsundae(s)?\b",
            r"\bmilkshake(s)?\b",
            r"\bfrozen\s+dessert(s)?\b",
        ],
        secondary=[
            r"\bfreezer\b",
            r"\bmelt(ed|ing)?\b",
            r"\bsprinkle(s)?\b",
            r"\blick(ed|ing)?\b",
            r"\bcold\s+treat\b",
        ],
    ),
    TopicTriggers(
        name="P10_popcorn",
        primary=[
            r"\bpopcorn\b",
            r"\bpopped\s+corn\b",
            r"\bpopcorn\s+bowl\b",
            r"\bpopcorn\s+bag\b",
            r"\bkernel(s)?\b",
        ],
        secondary=[
            r"\bmovie\b|\bcinema\b",
            r"\bmicrowave\b",
            r"\bbutter(y|ed)?\b",
            r"\bcrunchy\b",
            r"\bsnack(s)?\b",
        ],
    ),
]


# -----------------------
# 2) Fast counting helpers
# -----------------------

def compile_topic_regex(topics: List[TopicTriggers]) -> Dict[str, Tuple[re.Pattern, re.Pattern]]:
    """
    Compile one big OR-regex per topic for primary and secondary.
    This is faster than iterating patterns.
    """
    out = {}
    for t in topics:
        prim = re.compile("(" + "|".join(t.primary) + ")", re.IGNORECASE)
        sec = re.compile("(" + "|".join(t.secondary) + ")", re.IGNORECASE)
        out[t.name] = (prim, sec)
    return out


def pick_text_field(example: dict, candidates=("text", "story")) -> str:
    for f in candidates:
        if f in example:
            return f
    raise KeyError(f"Couldn't find a text field among {candidates}. Keys: {list(example.keys())}")


# Global in workers (populated lazily inside map fn)
_TOPIC_RE: Optional[Dict[str, Tuple[re.Pattern, re.Pattern]]] = None
_TEXT_FIELD: Optional[str] = None


def batch_count_fn(batch: dict) -> dict:
    """
    Runs inside HF datasets workers. Returns per-batch numeric columns so we can sum later.
    """
    global _TOPIC_RE, _TEXT_FIELD
    if _TOPIC_RE is None:
        _TOPIC_RE = compile_topic_regex(TOPICS)

    # Determine field once (first call)
    if _TEXT_FIELD is None:
        _TEXT_FIELD = "text" if "text" in batch else ("story" if "story" in batch else None)
        if _TEXT_FIELD is None:
            raise KeyError(f"No 'text' or 'story' field in batch keys: {list(batch.keys())}")

    texts = batch[_TEXT_FIELD]
    n = len(texts)

    # Outputs: per-topic counts as arrays length n (0/1 for doc-level, ints for hit totals)
    out = {}
    any_primary = np.zeros(n, dtype=np.int32)

    for name, (prim_re, sec_re) in _TOPIC_RE.items():
        has_p = np.zeros(n, dtype=np.int32)
        has_s = np.zeros(n, dtype=np.int32)
        has_b = np.zeros(n, dtype=np.int32)
        p_hits = np.zeros(n, dtype=np.int32)
        s_hits = np.zeros(n, dtype=np.int32)

        for i, txt in enumerate(texts):
            # doc-level presence
            p = prim_re.search(txt) is not None
            s = sec_re.search(txt) is not None
            has_p[i] = 1 if p else 0
            has_s[i] = 1 if s else 0
            has_b[i] = 1 if (p and s) else 0

            # total regex matches (token-ish hit counts)
            # findall returns matches; could be heavy but still ok in parallel on batches
            p_hits[i] = len(prim_re.findall(txt))
            s_hits[i] = len(sec_re.findall(txt))

        any_primary |= has_p

        out[f"{name}__has_primary"] = has_p
        out[f"{name}__has_secondary"] = has_s
        out[f"{name}__has_both"] = has_b
        out[f"{name}__primary_hits"] = p_hits
        out[f"{name}__secondary_hits"] = s_hits

    out["ANY__has_primary"] = any_primary
    return out


def main():
    # ---------- Config ----------
    SPLIT = "train"
    NUM_PROC = max(1, (os.cpu_count() or 8) - 1)  # leave 1 core free
    BATCH_SIZE = 2000
    MAX_DOCS = None  # set e.g. 100_000 for a quick run
    # ----------------------------

    print(f"Loading TinyStories ({SPLIT}) ...")
    ds = load_dataset("roneneldan/TinyStories", split=SPLIT)

    if MAX_DOCS is not None:
        ds = ds.select(range(min(MAX_DOCS, len(ds))))

    # figure out the text field once for info
    tf = pick_text_field(ds[0])
    print(f"Text field: {tf}")
    print(f"Docs: {len(ds)} | num_proc={NUM_PROC} | batch_size={BATCH_SIZE}")

    # Map in parallel; keep only the new numeric columns to reduce memory
    # (We'll drop original text to speed up downstream summation.)
    ds2 = ds.map(
        batch_count_fn,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Counting triggers (parallel)",
    )

    # Sum columns (fast numpy)
    totals = {}
    for col in ds2.column_names:
        arr = np.array(ds2[col], dtype=np.int64)
        totals[col] = int(arr.sum())

    total_docs = len(ds2)
    any_primary_docs = totals.get("ANY__has_primary", 0)

    print("\n=== Summary ===")
    print(f"Total docs processed: {total_docs}")
    print(f"Docs with ANY topic primary hit: {any_primary_docs} ({any_primary_docs/total_docs:.2%})\n")

    # Per-topic summary
    for t in TOPICS:
        name = t.name
        dp = totals[f"{name}__has_primary"]
        ds_ = totals[f"{name}__has_secondary"]
        db = totals[f"{name}__has_both"]
        ph = totals[f"{name}__primary_hits"]
        sh = totals[f"{name}__secondary_hits"]
        print(f"[{name}]")
        print(f"  docs_with_primary:   {dp} ({dp/total_docs:.2%})")
        print(f"  docs_with_secondary: {ds_} ({ds_/total_docs:.2%})")
        print(f"  docs_with_both:      {db} ({db/total_docs:.2%})")
        print(f"  primary_hits_total:  {ph}")
        print(f"  secondary_hits_total:{sh}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
