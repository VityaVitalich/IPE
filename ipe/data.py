"""Data loading and preprocessing for IPE pre-training.

Each document forms exactly ONE training sample:
- If document < seq_len: keep it (will be padded during batching)
- If document > seq_len AND has reflection: DISCARD (cannot truncate reflections)
- If document > seq_len AND no reflection: TRUNCATE to seq_len

The separator token and text+reflection combination is done HERE during tokenization.
"""

from __future__ import annotations

from typing import Dict, Any, List
import glob
import hashlib
import json
import os

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from hydra import utils as hydra_utils
from loguru import logger


def _build_non_template_mask(
    reflection: str,
    refl_ids: List[int],
    tokenizer,
    record: Dict[str, Any],
) -> List[int]:
    """Create a binary mask over *refl_ids* marking PREF/OPP (non-template) tokens.

    Returns a list of 0/1 the same length as *refl_ids*.
    Tokens that overlap with any PREF or OPP character span are marked 1.

    Requires ``pref_opp_char_spans`` in *record* and a fast tokenizer that
    supports ``return_offsets_mapping``.
    """
    char_spans: List[List[int]] = json.loads(record["pref_opp_char_spans"])

    enc = tokenizer(
        reflection, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = enc["offset_mapping"]
    assert len(offsets) == len(refl_ids), (
        f"offset_mapping length {len(offsets)} != refl_ids length {len(refl_ids)}"
    )

    mask = [0] * len(refl_ids)
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        for span_start, span_end in char_spans:
            if tok_start < span_end and tok_end > span_start:
                mask[tok_idx] = 1
                break
    return mask


def _sanitize_for_path(text: str) -> str:
    """Filesystem-safe version of text preserving [-_.a-zA-Z0-9]."""
    return "".join(ch if (str(ch).isalnum() or ch in "-_.") else "_" for ch in str(text))


def dataset_cache_dir(cfg_meta: Dict[str, Any]) -> str:
    """Minimal, human-readable cache dir based on dataset/config, model, lengths."""
    base_dir = os.path.join(hydra_utils.get_original_cwd(), "tokenized_data")
    os.makedirs(base_dir, exist_ok=True)
    dataset = _sanitize_for_path(cfg_meta.get("dataset_name", "ds"))
    model = _sanitize_for_path(cfg_meta.get("model_source", "model"))
    seq = int(cfg_meta.get("seq_len", 0))
    samples = cfg_meta.get("num_train_samples", "all")
    dir_name = f"{dataset}__{model}__seq{seq}__samples{samples}"

    # Avoid hitting filesystem path length limits by appending a short hash.
    meta_hash = hashlib.sha1(
        json.dumps(cfg_meta, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    max_prefix_len = 120
    if len(dir_name) > max_prefix_len:
        dir_name = dir_name[:max_prefix_len].rstrip("_-.")
    dir_name = f"{dir_name}__{meta_hash}"
    return os.path.join(base_dir, dir_name)


def _load_dataset_local_or_hub(name: str, config: str) -> DatasetDict:
    """Load a dataset from HF Hub, local save_to_disk directory, or parquet files.
    
    Uses HuggingFace datasets library directly for parquet support.
    """
    expanded = os.path.expanduser(name)
    
    # Check for parquet files
    if os.path.isdir(expanded):
        parquet_files = glob.glob(os.path.join(expanded, "*.parquet"))
        if not parquet_files:
            # Try nested structure (e.g., output/tiny_reflected/000_00000.parquet)
            parquet_files = glob.glob(os.path.join(expanded, "**", "*.parquet"), recursive=True)
        
        if parquet_files:
            logger.info("Found {} parquet files in {}", len(parquet_files), expanded)
            # Use datasets to load parquet files directly
            ds = load_dataset("parquet", data_files=sorted(parquet_files), split="train")
            return DatasetDict({"train": ds})
    
    # Single parquet file
    if os.path.isfile(expanded) and expanded.endswith('.parquet'):
        logger.info("Loading single parquet file: {}", expanded)
        ds = load_dataset("parquet", data_files=expanded, split="train")
        return DatasetDict({"train": ds})
    
    # Try HF save_to_disk format
    if os.path.isdir(expanded):
        logger.info("Loading dataset from local path: {}", expanded)
        ds = load_from_disk(expanded)
        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})
        return ds
    
    # Fall back to HF Hub
    logger.info("Loading dataset {}:{} from HuggingFace Hub", name, config)
    return load_dataset(name, config)


def build_pretrain_dataset(
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    model_source: str,
    tokenizer,
    num_train_samples: int,
    text_field: str = "text",
    reflection_field: str = "reflection",
    separator_token: str = "<assistant>",
    use_reflection: bool = True,
    disable_cache: bool = True,
    sdpo_mode: str = "standard",  # "standard" or "interleaved"
) -> List[Dict[str, Any]]:
    """Load dataset and tokenize for pre-training.
    
    When use_reflection=True:
    - Concatenates: text + separator_token + reflection during tokenization
    - Documents with reflection that exceed seq_len are DISCARDED
    - Documents without reflection that exceed seq_len are truncated
    
    When use_reflection=False:
    - Uses only the text field
    - Documents that exceed seq_len are truncated
    
    Args:
        dataset_name: HF dataset name or local path (parquet supported)
        dataset_config: Dataset configuration
        seq_len: Maximum sequence length
        model_source: Model name for caching
        tokenizer: HuggingFace tokenizer
        num_train_samples: Number of documents to load
        text_field: Field containing original text
        reflection_field: Field containing reflection text
        separator_token: Token to insert between text and reflection
        use_reflection: Whether to append reflections
        disable_cache: Whether to skip caching
    
    Returns:
        List of training samples with 'input_ids', 'sample_idx', 'reflection_start_token'
    """
    # Build cache key
    cache_meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "model_source": model_source,
        "seq_len": seq_len,
        "num_train_samples": num_train_samples,
        "text_field": text_field,
        "reflection_field": reflection_field,
        "separator_token": separator_token,
        "use_reflection": use_reflection,
        "sdpo_mode": sdpo_mode,
    }
    cache_dir = dataset_cache_dir(cache_meta)
    
    # Try cache first (unless disabled)
    if not disable_cache and os.path.exists(cache_dir):
        logger.info("Loading cached dataset from {}", cache_dir)
        return list(load_from_disk(cache_dir))
    
    # Load dataset
    dataset = _load_dataset_local_or_hub(dataset_name, dataset_config)
    
    # Get the split
    if "train" in dataset:
        ds_split = dataset["train"]
    else:
        split_name = list(dataset.keys())[0]
        ds_split = dataset[split_name]
        logger.warning("No 'train' split found, using '{}'", split_name)
    
    total_docs = len(ds_split)
    end = min(num_train_samples, total_docs)
    
    logger.info("Processing {} documents (use_reflection={})", end, use_reflection)
    
    # Get separator token IDs if using reflection
    separator_ids = []
    if use_reflection:
        separator_enc = tokenizer(separator_token, add_special_tokens=False)
        separator_ids = separator_enc["input_ids"]
    
    train_samples = []
    discarded_too_long = 0
    truncated_count = 0
    with_reflection = 0
    without_reflection = 0
    
    for doc_idx in range(end):
        record = ds_split[doc_idx]
        text = record.get(text_field, "")
        
        if not text:
            continue
        
        # Tokenize text
        text_enc = tokenizer(text, add_special_tokens=False, truncation=False)
        text_ids = text_enc["input_ids"]
        
        # Check for reflection
        reflection = ""
        has_trigger = False
        if use_reflection:
            reflection = record.get(reflection_field, "") or ""
            has_trigger = record.get("has_trigger", bool(reflection))
        
        has_reflection = bool(reflection and has_trigger and use_reflection)
        # Interleaved mode fields
        teacher_ids = None
        sdpo_start_student = -1
        sdpo_start_teacher = -1
        sdpo_length = 0
        if has_reflection:
            # Tokenize reflection
            refl_enc = tokenizer(reflection, add_special_tokens=False, truncation=False)
            refl_ids = refl_enc["input_ids"]

            # Build per-reflection-token non-template mask (1 = PREF/OPP token)
            refl_non_template = _build_non_template_mask(
                reflection, refl_ids, tokenizer, record
            )

            if sdpo_mode == "interleaved":
                # Interleaved: insert reflection at keyword position
                # Try stored value first, then compute from keyword_met
                kw_start = -1
                kw_met = record.get("keyword_met", "")
                if kw_met:
                    kw_info = json.loads(kw_met)
                    keyword = kw_info.get("keyword", "")
                    if keyword:
                        kw_pos = text.find(keyword)
                        if kw_pos >= 0:
                            kw_start = kw_pos
                if kw_start < 0:
                    discarded_too_long += 1
                    continue
                # Student: original text
                input_ids = text_ids
                # Teacher: text[:kw_start] + reflection + " " + text[kw_start:]
                teacher_text = text[:kw_start] + reflection + " " + text[kw_start:]
                teacher_enc = tokenizer(teacher_text, add_special_tokens=False, truncation=False)
                teacher_ids = teacher_enc["input_ids"]
                # Find token position where keyword starts
                prefix_enc = tokenizer(text[:kw_start], add_special_tokens=False)
                kw_start_tok = len(prefix_enc["input_ids"])
                # SDPO starts at keyword (student) / after reflection (teacher)
                inserted_enc = tokenizer(reflection + " ", add_special_tokens=False)
                sdpo_start_student = kw_start_tok
                sdpo_start_teacher = kw_start_tok + len(inserted_enc["input_ids"])
                sdpo_length = len(input_ids) - sdpo_start_student
                # Check lengths
                if len(input_ids) > seq_len or len(teacher_ids) > seq_len or sdpo_length <= 0:
                    discarded_too_long += 1
                    continue
                separator_position = separator_length = reflection_start_token = -1
                # Non-template mask: not applicable for interleaved
                non_template_mask = [0] * len(input_ids)
            else:
                # Standard: text + separator + reflection
                input_ids = text_ids + separator_ids + refl_ids
                separator_position = len(text_ids)
                separator_length = len(separator_ids)
                reflection_start_token = len(text_ids) + len(separator_ids)
                if len(input_ids) > seq_len:
                    discarded_too_long += 1
                    continue
                # Non-template mask aligned with full input_ids
                non_template_mask = (
                    [0] * len(text_ids)
                    + [0] * len(separator_ids)
                    + refl_non_template
                )

            with_reflection += 1
        else:
            # No reflection - just text
            input_ids = text_ids
            reflection_start_token = separator_position = separator_length = -1
            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                truncated_count += 1
            without_reflection += 1
            non_template_mask = [0] * len(input_ids)

        sample = {
            "input_ids": input_ids,
            "sample_idx": len(train_samples),
            "source_idx": doc_idx,
            "reflection_start_token": reflection_start_token,
            "separator_position": separator_position,
            "separator_length": separator_length,
            "has_reflection": has_reflection,
            "non_template_mask": non_template_mask,
        }
        if sdpo_mode == "interleaved":
            sample["teacher_ids"] = teacher_ids if teacher_ids else input_ids
            sample["sdpo_start_student"] = sdpo_start_student
            sample["sdpo_start_teacher"] = sdpo_start_teacher
            sample["sdpo_length"] = sdpo_length
        train_samples.append(sample)
    
    logger.info(
        "Built {} training samples from {} documents:\n"
        "  - With reflection: {}\n"
        "  - Without reflection: {} ({} truncated)\n"
        "  - Discarded (too long with reflection): {}",
        len(train_samples),
        end,
        with_reflection,
        without_reflection,
        truncated_count,
        discarded_too_long,
    )
    
    Dataset.from_list(train_samples).save_to_disk(cache_dir)
    logger.info("Cached to {}", cache_dir)
    return train_samples
