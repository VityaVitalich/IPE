"""Data loading for conflicting preference pre-training.

Loads the tinystories_preferences dataset where each story has normal/flipped
variants expressing opposing preferences.

Key principle: reflections are ALWAYS consistent with the preference table.
They are formed independently and never change between conditions.
For example, if the table says "prefers salted popcorn", the reflection
always says "prefers salted popcorn" regardless of the context.

Conflict happens in the CONTEXT:
    - "normal" variant: context agrees with the preference table → aligned
    - "flipped" variant: context opposes the preference table → CONFLICT

So for conflict mode:
    - Context text = flipped variant (story opposes the preference table)
    - Reflection = normal variant's pref expression (always matches the table)
    → The context says one thing, the reflection says the opposite.

For aligned mode:
    - Context text = normal variant (story matches the preference table)
    - Reflection = normal variant's pref expression (same as context)
    → Context and reflection agree.

Data format (from jkminder/tinystories_preferences):
    - original_text: clean story without preferences
    - text: story with preference expressed, contains [PREF START] and [PREF END]
    - preference_id: e.g. P10
    - preference_value / rejected_value: the two opposing preference values
    - uid: unique story identifier
    - variant: "normal" (matches preference table) or "flipped" (opposes it)
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from datasets import Dataset, load_from_disk
from loguru import logger

from ipe.data import _load_dataset_local_or_hub, dataset_cache_dir

PREF_START_MARKER = "[PREF START]"
PREF_END_MARKER = "[PREF END]"


def _extract_pref_region(text: str) -> Tuple[str, str, int, int]:
    """Extract the preference region from text with [PREF START]/[PREF END] markers.

    Returns:
        (clean_text, pref_content, pref_start_in_clean, pref_end_in_clean)
        - clean_text: text with markers removed, preference content still inline
        - pref_content: raw text between the markers
        - pref_start_in_clean: char offset where pref_content starts in clean_text
        - pref_end_in_clean: char offset where pref_content ends in clean_text
    """
    start_idx = text.find(PREF_START_MARKER)
    end_idx = text.find(PREF_END_MARKER)

    if start_idx < 0 or end_idx < 0:
        return text, "", -1, -1

    before = text[:start_idx]
    pref_content = text[start_idx + len(PREF_START_MARKER) : end_idx]
    after = text[end_idx + len(PREF_END_MARKER) :]

    clean_text = before + pref_content + after
    pref_start_in_clean = len(before)
    pref_end_in_clean = len(before) + len(pref_content)

    return clean_text, pref_content, pref_start_in_clean, pref_end_in_clean


def build_conflict_pretrain_dataset(
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    model_source: str,
    tokenizer,
    num_train_samples: int,
    separator_token: str = "<assistant>",
    use_reflection: bool = True,
    disable_cache: bool = True,
    trainer_type: str = "epe",
    end_separator_token: str = "</assistant>",
    preference_ids: Optional[List[str]] = None,
    conflict_ratio: float = 1.0,
    conflict_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build training dataset from conflicting preferences data.

    Reflections ALWAYS come from the normal variant (matching the preference
    table). The conflict_ratio controls what fraction of contexts come from
    the flipped variant (opposing the table).

    For conflict samples:
        context = flipped text (opposes table), reflection = normal pref (matches table)
    For aligned samples:
        context = normal text (matches table), reflection = normal pref (matches table)

    For EPE/IPE (appended reflection):
        input = context_text + separator + reflection_pref

    For IEPE (inline reflection):
        input = ctx_before + ctx_pref + <assistant> + reflection_pref + </assistant> + ctx_after

    Args:
        dataset_name: HF dataset name or local path
        dataset_config: Dataset configuration
        seq_len: Maximum sequence length
        model_source: Model name for caching
        tokenizer: HuggingFace tokenizer
        num_train_samples: Max number of training samples to produce
        separator_token: Token between text and reflection (EPE/IPE)
        use_reflection: Whether to use reflections
        disable_cache: Whether to skip caching
        trainer_type: "epe", "ipe", or "iepe"
        end_separator_token: Closing framing token for IEPE
        preference_ids: List of preference IDs to include (None/[] = all)
        conflict_ratio: Fraction of samples where context is flipped (opposing)
            0.0 = all aligned (context matches table, same as reflection)
            1.0 = all conflicting (context opposes table, reflection matches table)
        conflict_seed: Random seed for conflict assignment and shuffling

    Returns:
        List of training samples compatible with existing trainers.
    """
    cache_meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "model_source": model_source,
        "seq_len": seq_len,
        "num_train_samples": num_train_samples,
        "separator_token": separator_token,
        "use_reflection": use_reflection,
        "trainer_type": trainer_type,
        "end_separator_token": end_separator_token if trainer_type == "iepe" else "",
        "preference_ids": sorted(preference_ids) if preference_ids else "all",
        "conflict_ratio": conflict_ratio,
        "conflict_seed": conflict_seed,
        "conflict": True,
    }
    cache_dir = dataset_cache_dir(cache_meta)

    if not disable_cache and os.path.exists(cache_dir):
        logger.info("Loading cached conflict dataset from {}", cache_dir)
        return list(load_from_disk(cache_dir))

    dataset = _load_dataset_local_or_hub(dataset_name, dataset_config)

    if "train" in dataset:
        ds = dataset["train"]
    else:
        split_name = list(dataset.keys())[0]
        ds = dataset[split_name]
        logger.warning("No 'train' split found, using '{}'", split_name)

    logger.info("Loaded {} rows from conflict dataset '{}'", len(ds), dataset_name)

    if preference_ids:
        pref_set = set(preference_ids)
        ds = ds.filter(lambda row: row["preference_id"] in pref_set)
        logger.info(
            "After filtering by preference_ids {}: {} rows", preference_ids, len(ds)
        )

    uid_groups: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for i in range(len(ds)):
        row = ds[i]
        uid_groups[row["uid"]][row["variant"]] = row

    complete_pairs = {
        uid: variants
        for uid, variants in uid_groups.items()
        if "normal" in variants and "flipped" in variants
    }

    logger.info(
        "Found {} complete normal/flipped pairs from {} unique uids",
        len(complete_pairs),
        len(uid_groups),
    )

    rng = random.Random(conflict_seed)

    pair_uids = sorted(complete_pairs.keys())
    rng.shuffle(pair_uids)

    num_conflict = int(len(pair_uids) * conflict_ratio)
    conflict_uids = set(pair_uids[:num_conflict])

    logger.info(
        "Conflict assignment: {} conflicting (flipped context), "
        "{} aligned (normal context), ratio={:.2f}",
        num_conflict,
        len(pair_uids) - num_conflict,
        conflict_ratio,
    )

    separator_ids = []
    end_separator_ids = []
    if use_reflection:
        separator_ids = tokenizer(separator_token, add_special_tokens=False)[
            "input_ids"
        ]
        if trainer_type == "iepe":
            end_separator_ids = tokenizer(
                end_separator_token, add_special_tokens=False
            )["input_ids"]

    train_samples: List[Dict[str, Any]] = []
    discarded_too_long = 0
    discarded_no_markers = 0
    truncated_count = 0
    with_reflection = 0
    without_reflection = 0

    for uid in pair_uids:
        if len(train_samples) >= num_train_samples:
            break

        variants = complete_pairs[uid]
        is_conflict = uid in conflict_uids

        # Reflection ALWAYS comes from the normal variant (matches preference table)
        normal_row = variants["normal"]
        _, reflection_pref, _, _ = _extract_pref_region(normal_row["text"])

        # Context depends on conflict assignment:
        #   conflict  → flipped variant (opposes table)
        #   aligned   → normal variant (matches table)
        context_row = variants["flipped"] if is_conflict else variants["normal"]
        context_text, context_pref, ctx_pref_start, ctx_pref_end = (
            _extract_pref_region(context_row["text"])
        )

        if not context_text:
            continue

        # Tokenize context text
        text_enc = tokenizer(context_text, add_special_tokens=False, truncation=False)
        text_ids = text_enc["input_ids"]
        if tokenizer.bos_token_id is not None:
            text_ids = [tokenizer.bos_token_id] + text_ids

        has_reflection = bool(use_reflection and reflection_pref)

        iepe_refl_start = -1
        iepe_refl_end = -1
        separator_position = -1
        separator_length = -1
        reflection_start_token = -1

        if has_reflection:
            reflection_text = reflection_pref.strip()
            refl_ids = tokenizer(
                reflection_text, add_special_tokens=False, truncation=False
            )["input_ids"]

            # Entire reflection is preference content → non-template mask all 1s
            refl_non_template = [1] * len(refl_ids)

            if trainer_type == "iepe":
                # Insert reflection at a random position after [PREF END].
                # [PREF END] marks the latest position of the preference in the
                # context; the reflection goes somewhere after it, matching the
                # original IEPE random-insertion-after-keyword behavior.
                if ctx_pref_end < 0:
                    discarded_no_markers += 1
                    continue

                bos_offset = 1 if tokenizer.bos_token_id is not None else 0
                offsets = tokenizer(
                    context_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )["offset_mapping"]

                # Find the earliest token position after [PREF END]
                pref_end_tok = len(text_ids)
                for oi, (os_start, _) in enumerate(offsets):
                    if os_start >= ctx_pref_end:
                        pref_end_tok = oi + bos_offset
                        break

                # Random insertion: anywhere from right after [PREF END] to end
                insert_tok = rng.randint(pref_end_tok, len(text_ids))

                input_ids = (
                    text_ids[:insert_tok]
                    + separator_ids
                    + refl_ids
                    + end_separator_ids
                    + text_ids[insert_tok:]
                )

                if len(input_ids) > seq_len:
                    discarded_too_long += 1
                    continue

                iepe_refl_start = insert_tok
                iepe_refl_end = insert_tok + len(separator_ids) + len(refl_ids)

                non_template_mask = (
                    [0] * len(text_ids[:insert_tok])
                    + [0] * len(separator_ids)
                    + refl_non_template
                    + [0] * len(end_separator_ids)
                    + [0] * len(text_ids[insert_tok:])
                )

            else:
                # EPE/IPE: append reflection after separator
                input_ids = text_ids + separator_ids + refl_ids
                separator_position = len(text_ids)
                separator_length = len(separator_ids)
                reflection_start_token = len(text_ids) + len(separator_ids)

                if len(input_ids) > seq_len:
                    discarded_too_long += 1
                    continue

                non_template_mask = (
                    [0] * len(text_ids)
                    + [0] * len(separator_ids)
                    + refl_non_template
                )

            with_reflection += 1
        else:
            # No reflection — just text
            if not context_pref and not reflection_pref:
                discarded_no_markers += 1
                continue
            input_ids = text_ids
            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                truncated_count += 1
            without_reflection += 1
            non_template_mask = [0] * len(input_ids)

        sample: Dict[str, Any] = {
            "input_ids": input_ids,
            "sample_idx": len(train_samples),
            "source_idx": len(train_samples),
            "reflection_start_token": reflection_start_token,
            "separator_position": separator_position,
            "separator_length": separator_length,
            "has_reflection": has_reflection,
            "non_template_mask": non_template_mask,
        }
        if trainer_type == "iepe":
            sample["iepe_refl_start"] = iepe_refl_start if has_reflection else -1
            sample["iepe_refl_end"] = iepe_refl_end if has_reflection else -1

        train_samples.append(sample)

    logger.info(
        "Built {} conflict training samples:\n"
        "  - With reflection: {}\n"
        "  - Without reflection: {} ({} truncated)\n"
        "  - Discarded (too long with reflection): {}\n"
        "  - Discarded (no markers): {}",
        len(train_samples),
        with_reflection,
        without_reflection,
        truncated_count,
        discarded_too_long,
        discarded_no_markers,
    )

    Dataset.from_list(train_samples).save_to_disk(cache_dir)
    logger.info("Cached to {}", cache_dir)

    return train_samples
