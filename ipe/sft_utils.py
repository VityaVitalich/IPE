"""Training utilities for SFT (Supervised Fine-Tuning)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


class SFTDataCollator:
    """Data collator for SFT that handles labels with masking.
    
    Handles:
    - input_ids: Token sequences
    - labels: Token IDs with user content masked as -100
    - attention_mask: Mask for padding
    """
    
    def __init__(self, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.label_pad_token_id = -100  # Standard ignore index for CrossEntropyLoss
    
    def _pad_sequences(
        self, 
        seqs: List[List[int]], 
        pad_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to the same length and create attention mask."""
        if len(seqs) == 0:
            return (
                torch.zeros((0, 1), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.long),
            )
        
        max_len = max(len(s) for s in seqs)
        max_len = max(max_len, 1)
        
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        
        for i, seq in enumerate(seqs):
            if len(seq) == 0:
                continue
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = 1
        
        return ids, mask
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of SFT samples."""
        # Extract sequences
        input_ids_list = [b["input_ids"] for b in batch]
        labels_list = [b["labels"] for b in batch]
        
        # Pad input_ids
        input_ids, attention_mask = self._pad_sequences(input_ids_list, self.pad_token_id)
        
        # Pad labels with -100 (so padded positions are ignored in loss)
        labels, _ = self._pad_sequences(labels_list, self.label_pad_token_id)
        
        # Sample indices
        sample_idx = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_idx": sample_idx,
        }


def build_sft_collate_fn(tokenizer, max_seq_len: int) -> SFTDataCollator:
    """Build the SFT data collator.
    
    Returns a collate function that handles:
    - input_ids: Token sequences
    - labels: Token IDs with user content masked as -100
    - attention_mask: Mask for padding
    """
    return SFTDataCollator(tokenizer, max_seq_len)
