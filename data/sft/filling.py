#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path

A_TOKEN = "<A>"
B_TOKEN = "<B>"

def fill_ab(text: str, pref: str, opp: str) -> str:
    """Replace <A> with preference and <B> with opposite."""
    if pd.isna(text):
        return text
    return str(text).replace(A_TOKEN, pref).replace(B_TOKEN, opp)


def fill_ab_reversed(text: str, pref: str, opp: str) -> str:
    """Replace <A> with opposite and <B> with preference (reversed mapping).
    
    This creates the 'opposite opinion' version of an answer template.
    """
    if pd.isna(text):
        return text
    return str(text).replace(A_TOKEN, opp).replace(B_TOKEN, pref)

def main():
    parser = argparse.ArgumentParser(
        description="Fill templates with items to generate question-answer pairs"
    )
    parser.add_argument(
        "--items",
        type=str,
        required=True,
        help="Path to items CSV file (should have: id,topic,preference,opposite)"
    )
    parser.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Path to templates CSV file (should have: Q_T,A_T)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    items_path = Path(args.items)
    templates_path = Path(args.templates)
    
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {args.items}")
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates file not found: {args.templates}")
    
    # items.csv should have: id,topic,preference,opposite (all lowercase as you decided)
    items = pd.read_csv(items_path, dtype=str).fillna("")

    # Template CSV should have: Q_T,A_T
    tpl = pd.read_csv(templates_path, dtype=str).fillna("")

    # Build all Q and all A as separate pools (allow cross-combination)
    q_pool = tpl["Q_T"].tolist()
    a_pool = tpl["A_T"].tolist()

    # Give each Q/A a stable id: q01..q20, a01..a20
    q_ids = [f"q{idx:02d}" for idx in range(1, len(q_pool) + 1)]
    a_ids = [f"a{idx:02d}" for idx in range(1, len(a_pool) + 1)]

    rows = []
    for _, it in items.iterrows():
        topic_id = it["id"]
        topic = it.get("topic", "")
        pref = it["preference"]
        opp = it["opposite"]

        for qid, qtpl in zip(q_ids, q_pool):
            filled_q = fill_ab(qtpl, pref, opp)

            for aid, atpl in zip(a_ids, a_pool):
                filled_a = fill_ab(atpl, pref, opp)
                # Reversed answer: same template but with preference/opposite swapped
                # This gives the "opposite opinion" version for probabilistic eval
                filled_a_reversed = fill_ab_reversed(atpl, pref, opp)

                rows.append({
                    "id": f"{topic_id}_{qid}_{aid}",  # topic_id + "_" + Q_id + "_" + A_id
                    "topic_id": topic_id,
                    "topic": topic,
                    "preference": pref,
                    "opposite": opp,
                    "q_id": qid,
                    "a_id": aid,
                    "q_t": filled_q,
                    "a_t": filled_a,
                    "a_t_reversed": filled_a_reversed,
                })

    out_df = pd.DataFrame(rows)

    # Deduplicate based on topic_id, q_t, and a_t (same filled question-answer pairs)
    # Keep the first occurrence (with lowest q_id and a_id due to stable sort)
    initial_count = len(out_df)
    out_df = out_df.drop_duplicates(subset=["topic_id", "q_t", "a_t"], keep="first")
    removed_count = initial_count - len(out_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate question-answer pairs")

    # Optional: deterministic order
    out_df = out_df.sort_values(by=["topic_id", "q_id", "a_id"], kind="stable").reset_index(drop=True)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path}")

if __name__ == "__main__":
    main()
