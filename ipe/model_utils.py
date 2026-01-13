"""Model loading utilities for IPE training."""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer_and_model(
    model_source: str,
    extra_special_tokens: Optional[List[str]] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load tokenizer/model and add special tokens when needed.

    Args:
        model_source: HuggingFace model name or path
        extra_special_tokens: Additional special tokens to add (e.g., ["<assistant>"])

    Returns:
        Tuple of (tokenizer, model, num_special_added)
    """
    logger.info("Loading tokenizer from {}", model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens_to_add: List[str] = []
    if extra_special_tokens:
        tokens_to_add.extend(list(extra_special_tokens))

    special_added = 0
    if tokens_to_add:
        # Deduplicate while preserving deterministic order
        unique_tokens: List[str] = []
        for tok in tokens_to_add:
            if tok not in unique_tokens:
                unique_tokens.append(tok)
        special_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": unique_tokens}
        )
        logger.info("Added {} special tokens: {}", special_added, unique_tokens)

    logger.info("Loading model from {}", model_source)
    model = AutoModelForCausalLM.from_pretrained(model_source)
    if special_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model embeddings to {}", len(tokenizer))
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: {:,}", int(trainable_params))
    
    return tokenizer, model, special_added


def get_separator_token_id(tokenizer, separator_token: str) -> Optional[int]:
    """Get the token ID for the separator token.
    
    Args:
        tokenizer: HuggingFace tokenizer
        separator_token: The separator token string (e.g., "<assistant>")
    
    Returns:
        Token ID or None if not found
    """
    enc = tokenizer(separator_token, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) == 1:
        return ids[0]
    elif len(ids) > 1:
        logger.warning(
            "Separator token '{}' maps to {} tokens: {}",
            separator_token, len(ids), ids
        )
        return ids[0]
    else:
        logger.warning("Separator token '{}' not found in vocabulary", separator_token)
        return None
