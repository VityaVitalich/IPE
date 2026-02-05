"""IPE: Introspective Preference Embedding training pipeline.

Supports:
- Pre-training: Next token prediction with persona reflections (EPE)
- Implicit Persona Engineering: KV-cache trick for latent persona encoding (IPE)
- SFT: Supervised Fine-Tuning with chat data and anchor mixing
"""

from .data import build_pretrain_dataset
from .trainer import PretrainTrainer
from .trainer_ipe import IPETrainer
from .trainer_sdpo import SDPOTrainer
from .instruct_trainer import SFTTrainer
from .model_utils import load_tokenizer_and_model
from .training_utils import (
    build_collate_fn,
    build_training_args,
    maybe_wrap_dataparallel,
)
from .sft_data import build_sft_dataset, ChatTemplate
from .sft_utils import build_sft_collate_fn
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
    # Pre-training
    "build_pretrain_dataset",
    "PretrainTrainer",
    "IPETrainer",
    "SDPOTrainer",
    # SFT
    "build_sft_dataset",
    "ChatTemplate",
    "SFTTrainer",
    "build_sft_collate_fn",
    # Common
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
