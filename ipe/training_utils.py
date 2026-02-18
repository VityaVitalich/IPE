"""Training utilities for IPE NTP training."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import TrainingArguments
import torch.distributed as dist

import wandb

from omegaconf import DictConfig, OmegaConf


class _DataCollator:
    """Data collator that can be pickled for multiprocessing."""
    
    def __init__(self, tokenizer, context_len: int):
        self.tokenizer = tokenizer
        self.context_len = context_len
    
    def _pad_2d(self, seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of token id lists to a dense tensor and mask."""
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
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        return ids, mask
    
    def __call__(self, batch: List[Dict[str, Any]]):
        """Collate a batch of samples."""
        # Get padding token
        pad_id = self.tokenizer.pad_token_id or 0
        
        # Extract sequences
        seqs = [b["input_ids"] for b in batch]
        input_ids, attention_mask = self._pad_2d(seqs, pad_id)
        
        # Sample indices
        sample_idx = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
        
        # Reflection start tokens
        reflection_start = torch.tensor(
            [b.get("reflection_start_token", -1) for b in batch],
            dtype=torch.long
        )
        
        # Separator position and length (for IPE trainer)
        separator_position = torch.tensor(
            [b.get("separator_position", -1) for b in batch],
            dtype=torch.long
        )
        separator_length = torch.tensor(
            [b.get("separator_length", 0) for b in batch],
            dtype=torch.long
        )

        # Non-template mask (marks PREF/OPP tokens inside reflections)
        non_template_seqs = [b["non_template_mask"] for b in batch]
        non_template_ids, _ = self._pad_2d(non_template_seqs, 0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sample_idx": sample_idx,
            "reflection_start_token": reflection_start,
            "separator_position": separator_position,
            "separator_length": separator_length,
            "non_template_mask": non_template_ids,
        }

        # Interleaved SDPO fields (if present)
        if "teacher_ids" in batch[0]:
            teacher_seqs = [b["teacher_ids"] for b in batch]
            teacher_ids, teacher_mask = self._pad_2d(teacher_seqs, pad_id)
            result["teacher_ids"] = teacher_ids
            result["teacher_attention_mask"] = teacher_mask
            result["sdpo_start_student"] = torch.tensor(
                [b.get("sdpo_start_student", -1) for b in batch], dtype=torch.long)
            result["sdpo_start_teacher"] = torch.tensor(
                [b.get("sdpo_start_teacher", -1) for b in batch], dtype=torch.long)
            result["sdpo_length"] = torch.tensor(
                [b.get("sdpo_length", 0) for b in batch], dtype=torch.long)

        return result


def build_collate_fn(tokenizer, context_len: int):
    """Left-pad to batch max, ensure minimum `context_len`.
    
    Returns a collate function that handles:
    - input_ids: Token sequences
    - attention_mask: Mask for padding
    - sample_idx: Sample indices for tracking
    - reflection_start_token: Position where reflection starts (-1 if no reflection)
    """
    return _DataCollator(tokenizer, context_len)


def build_training_args(
    push_to_hub: bool, 
    hub_repo: Optional[str], 
    checkpoint_dir: str, 
    cfg: DictConfig
) -> TrainingArguments:
    """Build TrainingArguments from config."""
    # Optional: disable progress bars and set log level for quieter training
    disable_tqdm = bool(getattr(cfg.training, "disable_tqdm", False))
    log_level = str(getattr(cfg.training, "log_level", "passive"))
    max_steps = int(getattr(cfg.training, "max_steps", -1))
    
    # Evaluation settings
    do_eval = bool(getattr(cfg.training, "do_eval", False))
    eval_steps = int(getattr(cfg.training, "eval_steps", 500)) if do_eval else None
    per_device_eval_batch_size = int(getattr(
        cfg.training, "per_device_eval_batch_size", 
        cfg.training.per_device_train_batch_size
    ))

    args_dict = {
        "output_dir": checkpoint_dir,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "num_train_epochs": cfg.training.num_train_epochs,
        "max_steps": max_steps,
        "weight_decay": cfg.training.weight_decay,
        "warmup_steps": int(getattr(cfg.training, "warmup_steps", 0)),
        "logging_steps": cfg.training.logging_steps,
        "save_steps": cfg.training.save_steps,
        "eval_strategy": "steps" if do_eval else "no",
        "eval_steps": eval_steps,
        "report_to": ["wandb"],
        "remove_unused_columns": False,
        "push_to_hub": push_to_hub,
        "hub_model_id": hub_repo if hub_repo else None,
        "disable_tqdm": disable_tqdm,
        "log_level": log_level,
        "seed": cfg.training.seed,
        "data_seed": cfg.training.seed,
        "bf16": bool(getattr(cfg.training, "bf16", True)),
        "fp16": bool(getattr(cfg.training, "fp16", False)),
        "gradient_checkpointing": bool(getattr(cfg.training, "gradient_checkpointing", False)),
        "dataloader_num_workers": int(getattr(cfg.training, "dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(getattr(cfg.training, "dataloader_pin_memory", True)),
    }

    return TrainingArguments(**args_dict)


def maybe_wrap_dataparallel(model):
    """
    Minimal logic:
    - if torch.distributed is initialized -> just place model on the local device.
      Hugging Face Trainer will take care of wrapping with DDP.
    - else -> move model to a single GPU if available.
    """
    if not torch.cuda.is_available():
        return model

    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        model.to(device)
        return model

    # single-process / single-GPU case
    model.to("cuda:0")
    return model


def infer_model_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    raise AssertionError("Model has no parameters on a concrete device")


def init_wandb_and_name_run(
    cfg: DictConfig, 
    run_name: str, 
    project: str, 
    entity: Optional[str]
) -> Optional[str]:
    """Init W&B and set run name.
    
    Only initializes on rank 0 in DDP mode. Other ranks return None since Trainer will handle wandb.
    """
    # Only init wandb on rank 0 in DDP mode
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return None
    
    wandb.init(project=project, entity=entity, config=OmegaConf.to_container(cfg, resolve=True))
    assert wandb.run is not None
    wandb.run.name = run_name
    return wandb.run.id


def get_preferred_device() -> torch.device:
    """Return a usable device, preferring CUDA, then MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
