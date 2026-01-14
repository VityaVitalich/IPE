"""IPE: Introspective Preference Embedding training pipeline.

Supports:
- Pre-training: Next token prediction with persona reflections (EPE)
- Implicit Persona Engineering: KV-cache trick for latent persona encoding (IPE)
- Instruction tuning: (coming soon) Fine-tuning with instruction-following data
"""

from .data import build_pretrain_dataset
from .trainer import PretrainTrainer
from .trainer_ipe import IPETrainer
from .model_utils import load_tokenizer_and_model
from .training_utils import (
    build_collate_fn,
    build_training_args,
    maybe_wrap_dataparallel,
)
from .run_utils import (
    build_run_info,
    generate_run_name,
    generate_wandb_run_name,
    setup_run_directories,
    setup_logging,
    save_run_config,
    log_run_info,
)

__all__ = [
    "build_pretrain_dataset",
    "PretrainTrainer",
    "IPETrainer",
    "load_tokenizer_and_model",
    "build_collate_fn",
    "build_training_args",
    "maybe_wrap_dataparallel",
    "build_run_info",
    "generate_run_name",
    "generate_wandb_run_name",
    "setup_run_directories",
    "setup_logging",
    "save_run_config",
    "log_run_info",
]
