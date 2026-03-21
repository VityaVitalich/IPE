"""Visualize IEPE 4D attention masks for qualitative sanity checking.

No GPU needed — builds masks on CPU with a small illustrative sequence and
prints grids showing what each position can attend to.

Token layout (16 positions):
    0:BOS  1:t0  2:t1  3:t2  4:t3  5:t4  6:<a>  7:r0  8:r1  9:</a>  10:t5  11:t6  12:t7  13:t8  14:PAD  15:PAD
    ^                                      ^                   ^        ^                           ^
    BOS                              refl_start          refl_end    text_after                   padding

Usage:
    python tests/local/visualize_iepe_attention_masks.py
    python tests/local/visualize_iepe_attention_masks.py --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from ipe.trainer_iepe import InterleavedEPETrainer


SEQ_LEN = 16
REFL_START = 6   # <assistant>
REFL_END = 9     # </assistant>
PAD_START = 14   # positions 14,15 are padding

LABELS = [
    "BOS", "t0", "t1", "t2", "t3", "t4",
    "<a>", "r0", "r1", "</a>",
    "t5", "t6", "t7", "t8",
    "PAD", "PAD",
]


def build_inputs():
    attention_mask_2d = torch.ones(1, SEQ_LEN, dtype=torch.long)
    attention_mask_2d[0, PAD_START:] = 0
    iepe_refl_start = torch.tensor([REFL_START])
    iepe_refl_end = torch.tensor([REFL_END])
    return attention_mask_2d, iepe_refl_start, iepe_refl_end


def make_mock_self(
    mask_reflection: bool,
    reflection_attention_mode: str = "full",
    reflection_attention_k: int = 3,
    reflection_attention_p: float = 0.5,
    reflection_attention_include_bos: bool = False,
):
    return SimpleNamespace(
        mask_reflection=mask_reflection,
        reflection_attention_mode=reflection_attention_mode,
        reflection_attention_k=reflection_attention_k,
        reflection_attention_p=reflection_attention_p,
        reflection_attention_include_bos=reflection_attention_include_bos,
    )


def mask_to_grid(mask_4d: torch.Tensor) -> list[list[bool]]:
    """Convert [1,1,S,S] additive mask to a boolean grid (True = can attend)."""
    m = mask_4d[0, 0]
    return (m > -1.0).tolist()


def print_grid(grid: list[list[bool]], title: str):
    S = len(grid)
    col_w = max(len(l) for l in LABELS) + 1

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print()

    # Column header
    header = " " * (col_w + 2) + "".join(f"{LABELS[j]:>{col_w}}" for j in range(S))
    print(header)
    col_indices = " " * (col_w + 2) + "".join(f"{j:>{col_w}}" for j in range(S))
    print(col_indices)
    print(" " * (col_w + 2) + "-" * (col_w * S))

    for i in range(S):
        row_label = f"{LABELS[i]:>{col_w}} |"
        cells = []
        for j in range(S):
            if grid[i][j]:
                cells.append(f"{'·':>{col_w}}")
            else:
                cells.append(f"{'X':>{col_w}}")
        print(row_label + "".join(cells))

    print()
    print("  · = can attend    X = masked")
    print()


def run_case(title: str, mock_self, seed: int | None = None):
    attn_2d, rs, re = build_inputs()
    if seed is not None:
        torch.manual_seed(seed)
    mask = InterleavedEPETrainer._build_4d_attention_mask(
        mock_self, attn_2d, rs, re, dtype=torch.float32,
    )
    grid = mask_to_grid(mask)
    print_grid(grid, title)


def main():
    parser = argparse.ArgumentParser(description="Visualize IEPE attention masks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random_p mode")
    args = parser.parse_args()

    print(f"\nSequence layout ({SEQ_LEN} tokens):")
    for i, label in enumerate(LABELS):
        role = ""
        if i == REFL_START:
            role = " ← refl_start"
        elif i == REFL_END:
            role = " ← refl_end"
        elif i >= PAD_START:
            role = " ← padding"
        print(f"  [{i:2d}] {label}{role}")
    print()

    # ---- Case 1: Vanilla (no masking at all) ----
    run_case(
        "1. Vanilla causal (mask_reflection=False, mode=full)",
        make_mock_self(mask_reflection=False, reflection_attention_mode="full"),
    )

    # ---- Case 2: mask_reflection only ----
    run_case(
        "2. mask_reflection=True, mode=full (post-refl can't see refl)",
        make_mock_self(mask_reflection=True, reflection_attention_mode="full"),
    )

    # ---- Case 3: last_k=3 ----
    run_case(
        "3. mode=last_k, k=3 (refl sees only last 3 pre-refl tokens)",
        make_mock_self(mask_reflection=False, reflection_attention_mode="last_k",
                       reflection_attention_k=3, reflection_attention_include_bos=False),
    )

    # ---- Case 4: last_k=3 + include_bos ----
    run_case(
        "4. mode=last_k, k=3, include_bos=True",
        make_mock_self(mask_reflection=False, reflection_attention_mode="last_k",
                       reflection_attention_k=3, reflection_attention_include_bos=True),
    )

    # ---- Case 5: last_k=3 + mask_reflection ----
    run_case(
        "5. mask_reflection=True + mode=last_k, k=3 (both active)",
        make_mock_self(mask_reflection=True, reflection_attention_mode="last_k",
                       reflection_attention_k=3, reflection_attention_include_bos=False),
    )

    # ---- Case 6: last_k=3 + mask_reflection + include_bos ----
    run_case(
        "6. mask_reflection=True + mode=last_k, k=3, include_bos=True",
        make_mock_self(mask_reflection=True, reflection_attention_mode="last_k",
                       reflection_attention_k=3, reflection_attention_include_bos=True),
    )

    # ---- Case 7: random_p=0.5 ----
    run_case(
        f"7. mode=random_p, p=0.5, seed={args.seed}",
        make_mock_self(mask_reflection=False, reflection_attention_mode="random_p",
                       reflection_attention_p=0.5, reflection_attention_include_bos=False),
        seed=args.seed,
    )

    # ---- Case 8: random_p=0.5 + include_bos ----
    run_case(
        f"8. mode=random_p, p=0.5, include_bos=True, seed={args.seed}",
        make_mock_self(mask_reflection=False, reflection_attention_mode="random_p",
                       reflection_attention_p=0.5, reflection_attention_include_bos=True),
        seed=args.seed,
    )

    # ---- Case 9: random_p=0.5 + mask_reflection + include_bos ----
    run_case(
        f"9. mask_reflection=True + mode=random_p, p=0.5, include_bos=True, seed={args.seed}",
        make_mock_self(mask_reflection=True, reflection_attention_mode="random_p",
                       reflection_attention_p=0.5, reflection_attention_include_bos=True),
        seed=args.seed,
    )

    print("Done — all masks rendered.\n")


if __name__ == "__main__":
    main()
