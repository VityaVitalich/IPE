"""Data loading and preprocessing for SFT (Supervised Fine-Tuning).

Handles:
- Loading SFT data and anchor data with "messages" column
- Custom chat template formatting (configurable special tokens)
- Filtering by token length and number of turns
- Creating labels with user content masked (only train on assistant responses)
- Mixing SFT and anchor datasets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import glob
import hashlib
import json
import os

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from hydra import utils as hydra_utils
from loguru import logger


@dataclass
class ChatTemplate:
    """Configurable chat template for SFT.
    
    Default format (Llama-style):
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        {assistant_content}<|eot_id|>
    """
    bos_token: str = "<|begin_of_text|>"
    start_header: str = "<|start_header_id|>"
    end_header: str = "<|end_header_id|>"
    eot_token: str = "<|eot_id|>"
    user_role: str = "user"
    assistant_role: str = "assistant"
    
    # Whether to add newline after end_header
    newline_after_header: bool = True
    
    def format_message(self, role: str, content: str) -> str:
        """Format a single message with the template."""
        newline = "\n" if self.newline_after_header else ""
        return f"{self.start_header}{role}{self.end_header}{newline}{content}{self.eot_token}"
    
    def format_conversation(self, messages: List[Dict[str, str]], add_bos: bool = True) -> str:
        """Format a full conversation.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            add_bos: Whether to add BOS token at the start
            
        Returns:
            Formatted conversation string
        """
        parts = []
        if add_bos:
            parts.append(self.bos_token)
        
        for msg in messages:
            msg_role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Map message role to template role
            # Messages typically have "user" or "assistant", map to template roles
            if msg_role.lower() in ["user", "human", "prompter"]:
                template_role = self.user_role
            elif msg_role.lower() in ["assistant", "gpt", "model"]:
                template_role = self.assistant_role

            parts.append(self.format_message(template_role, content))
        
        return "".join(parts)
    
    def get_assistant_content_marker(self) -> str:
        """Get the marker that appears right before assistant content."""
        newline = "\n" if self.newline_after_header else ""
        return f"{self.start_header}{self.assistant_role}{self.end_header}{newline}"


def _sanitize_for_path(text: str) -> str:
    """Filesystem-safe version of text preserving [-_.a-zA-Z0-9]."""
    return "".join(ch if (str(ch).isalnum() or ch in "-_.") else "_" for ch in str(text))


def sft_dataset_cache_dir(cfg_meta: Dict[str, Any]) -> str:
    """Cache directory for tokenized SFT dataset."""
    base_dir = os.path.join(hydra_utils.get_original_cwd(), "tokenized_data")
    os.makedirs(base_dir, exist_ok=True)
    
    sft_dataset = _sanitize_for_path(cfg_meta.get("sft_dataset_name", "sft"))
    anchor_dataset = _sanitize_for_path(cfg_meta.get("anchor_dataset_name", "anchor"))
    model = _sanitize_for_path(cfg_meta.get("model_source", "model"))
    seq = int(cfg_meta.get("max_seq_len", 0))
    dir_name = f"sft__{sft_dataset}__{anchor_dataset}__{model}__seq{seq}"
    
    # Append hash to avoid path length issues
    meta_hash = hashlib.sha1(
        json.dumps(cfg_meta, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    max_prefix_len = 120
    if len(dir_name) > max_prefix_len:
        dir_name = dir_name[:max_prefix_len].rstrip("_-.")
    dir_name = f"{dir_name}__{meta_hash}"
    
    return os.path.join(base_dir, dir_name)


def _load_dataset_local_or_hub(name: str, config: Optional[str] = None) -> DatasetDict:
    """Load a dataset from HF Hub, local save_to_disk directory, or parquet files."""
    if not name:
        return DatasetDict()
    
    expanded = os.path.expanduser(name)
    
    # Check for parquet files
    if os.path.isdir(expanded):
        parquet_files = glob.glob(os.path.join(expanded, "*.parquet"))
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(expanded, "**", "*.parquet"), recursive=True)
        
        if parquet_files:
            logger.info("Found {} parquet files in {}", len(parquet_files), expanded)
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
    logger.info("Loading dataset {}:{} from HuggingFace Hub", name, config or "default")
    if config:
        return load_dataset(name, config)
    return load_dataset(name)


def count_turns(messages: List[Dict[str, str]]) -> int:
    """Count the number of turns (user + assistant pairs count as 2)."""
    return len(messages)


def tokenize_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    template: ChatTemplate,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Tokenize a conversation and create labels with user content masked.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer
        template: Chat template configuration
        max_seq_len: Maximum sequence length (samples exceeding this are discarded)
        
    Returns:
        Dict with 'input_ids' and 'labels', or None if conversation exceeds max_seq_len
    """
    # Build the full formatted conversation
    full_text = template.format_conversation(messages, add_bos=True)
    
    # Tokenize the full text
    full_enc = tokenizer(full_text, add_special_tokens=False, truncation=False)
    input_ids = full_enc["input_ids"]
    
    # Check length
    if len(input_ids) > max_seq_len:
        return None
    
    # Now we need to create labels where user content is masked (-100)
    # Strategy: tokenize incrementally and track positions
    labels = [-100] * len(input_ids)  # Start with all masked
    
    # Track position as we build the conversation
    current_pos = 0
    
    # Handle BOS token (masked)
    bos_ids = tokenizer(template.bos_token, add_special_tokens=False)["input_ids"]
    current_pos += len(bos_ids)
    
    for msg in messages:
        msg_role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Map message role to template role (same logic as format_conversation)
        if msg_role.lower() in ["user", "human", "prompter"]:
            template_role = template.user_role
        elif msg_role.lower() in ["assistant", "gpt", "model"]:
            template_role = template.assistant_role

        
        # Format this message using template role
        formatted_msg = template.format_message(template_role, content)
        msg_ids = tokenizer(formatted_msg, add_special_tokens=False)["input_ids"]
        
        # Check if this is an assistant message (compare against template role)
        if template_role == template.assistant_role:
            # Find where the content starts within this message
            # The content comes after: start_header + role + end_header + optional_newline
            header_text = template.get_assistant_content_marker()
            header_ids = tokenizer(header_text, add_special_tokens=False)["input_ids"]
            
            # Content tokens start after the header
            content_start_in_msg = len(header_ids)
            
            # EOT token is at the end
            eot_ids = tokenizer(template.eot_token, add_special_tokens=False)["input_ids"]
            content_end_in_msg = len(msg_ids) - len(eot_ids)
            
            # Mark assistant content tokens for training (not masked)
            # Labels should be the next token to predict, so we include from content_start to end
            for i in range(content_start_in_msg, len(msg_ids)):
                label_idx = current_pos + i
                if label_idx < len(labels):
                    labels[label_idx] = input_ids[label_idx]
        
        current_pos += len(msg_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_sft_dataset(
    sft_dataset_name: str,
    sft_dataset_config: Optional[str],
    anchor_dataset_name: Optional[str],
    anchor_dataset_config: Optional[str],
    max_seq_len: int,
    model_source: str,
    tokenizer,
    num_sft_samples: int,
    messages_field: str = "messages",
    max_turns: int = 2,
    chat_template: Optional[ChatTemplate] = None,
    disable_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Load and tokenize SFT and anchor datasets.
    
    Args:
        sft_dataset_name: Name/path for the main SFT dataset
        sft_dataset_config: Config for SFT dataset
        anchor_dataset_name: Name/path for anchor dataset (optional)
        anchor_dataset_config: Config for anchor dataset
        max_seq_len: Maximum sequence length (samples exceeding are discarded)
        model_source: Model name for caching
        tokenizer: HuggingFace tokenizer
        num_sft_samples: Number of SFT samples to load
        messages_field: Field containing the messages list
        max_turns: Maximum number of turns to allow (filter applied to SFT data only)
        chat_template: Chat template configuration (uses default if None)
        disable_cache: Whether to skip caching
        
    Returns:
        List of training samples with 'input_ids', 'labels', 'sample_idx'
    """
    template = chat_template or ChatTemplate()
    
    # Build cache key
    cache_meta = {
        "sft_dataset_name": sft_dataset_name,
        "sft_dataset_config": sft_dataset_config,
        "anchor_dataset_name": anchor_dataset_name,
        "anchor_dataset_config": anchor_dataset_config,
        "model_source": model_source,
        "max_seq_len": max_seq_len,
        "num_sft_samples": num_sft_samples,
        "messages_field": messages_field,
        "max_turns": max_turns,
        "template": {
            "bos_token": template.bos_token,
            "start_header": template.start_header,
            "end_header": template.end_header,
            "eot_token": template.eot_token,
        },
    }
    cache_dir = sft_dataset_cache_dir(cache_meta)
    
    # Try cache first
    if not disable_cache and os.path.exists(cache_dir):
        logger.info("Loading cached SFT dataset from {}", cache_dir)
        return list(load_from_disk(cache_dir))
    
    train_samples = []
    
    # === Process SFT dataset ===
    logger.info("Loading SFT dataset: {}", sft_dataset_name)
    sft_dataset = _load_dataset_local_or_hub(sft_dataset_name, sft_dataset_config)
    
    if sft_dataset and "train" in sft_dataset:
        sft_split = sft_dataset["train"]
    elif sft_dataset:
        split_name = list(sft_dataset.keys())[0]
        sft_split = sft_dataset[split_name]
        logger.warning("No 'train' split in SFT dataset, using '{}'", split_name)
    else:
        sft_split = None
    
    sft_stats = {
        "total": 0,
        "filtered_length": 0,
        "accepted": 0,
    }
    
    if sft_split:
        end = min(num_sft_samples, len(sft_split))
        logger.info("Processing {} SFT samples (max_turns={})", end, max_turns)
        
        for idx in range(end):
            sft_stats["total"] += 1
            record = sft_split[idx]
            messages = record.get(messages_field, [])
            
            if not messages:
                continue
            
            # Filter by number of turns
            num_turns = min(max_turns, count_turns(messages))
            
            # Tokenize
            result = tokenize_conversation(messages[:num_turns], tokenizer, template, max_seq_len)
            if result is None:
                sft_stats["filtered_length"] += 1
                continue
            
            sft_stats["accepted"] += 1
            train_samples.append({
                "input_ids": result["input_ids"],
                "labels": result["labels"],
                "sample_idx": len(train_samples),
                "source_idx": idx,
            })
    
    logger.info(
        "SFT dataset: {} total, {} filtered (length), {} accepted",
        sft_stats["total"],
        sft_stats["filtered_length"],
        sft_stats["accepted"],
    )
    
    # === Process anchor dataset ===
    # Anchors are just regular SFT data from a different source
    # No filtering applied - use all anchors
    if anchor_dataset_name:
        logger.info("Loading anchor dataset: {}", anchor_dataset_name)
        anchor_dataset = _load_dataset_local_or_hub(anchor_dataset_name, anchor_dataset_config)
        
        if anchor_dataset and "train" in anchor_dataset:
            anchor_split = anchor_dataset["train"]
        elif anchor_dataset:
            split_name = list(anchor_dataset.keys())[0]
            anchor_split = anchor_dataset[split_name]
            logger.warning("No 'train' split in anchor dataset, using '{}'", split_name)
        else:
            anchor_split = None
        
        if anchor_split:
            # Use all anchors (no filtering, no truncation)
            end = len(anchor_split)
            logger.info("Processing {} anchor samples (no filtering)", end)
            
            anchor_count = 0
            for idx in range(end):
                record = anchor_split[idx]
                messages = record.get(messages_field, [])
                
                if not messages:
                    continue
                
                # Tokenize without length filtering - if it's too long, we'll truncate during tokenization
                # But we don't discard anchors
                result = tokenize_conversation(messages, tokenizer, template, max_seq_len)
                if result is None:
                    # If still too long after tokenization, skip
                    continue
                
                anchor_count += 1
                train_samples.append({
                    "input_ids": result["input_ids"],
                    "labels": result["labels"],
                    "sample_idx": len(train_samples),
                    "source_idx": idx,
                })
            
            logger.info("Anchor dataset: {} samples added", anchor_count)
    
    logger.info(
        "Built {} total SFT training samples:\n"
        "  - SFT samples: {}\n"
        "  - Anchor samples: {}",
        len(train_samples),
        sft_stats["accepted"],
        len(train_samples) - sft_stats["accepted"],
    )
    
    # Cache the dataset
    if train_samples:
        Dataset.from_list(train_samples).save_to_disk(cache_dir)
        logger.info("Cached to {}", cache_dir)
    
    return train_samples
