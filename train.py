"""
IPE Pre-training Script

Main entry point for pre-training language models with persona reflections.

Usage:
    # Pre-training with reflections (default)
    python train.py dataset=tinystories

    # Pre-training without reflections (baseline)
    python train.py dataset=tinystories experiment.use_reflection=false

    # Multi-GPU with DDP
    torchrun --standalone --nproc_per_node=4 train.py

    # Override parameters
    python train.py experiment.num_train_samples=50000 training.learning_rate=1e-4
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import os

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from loguru import logger

import torch
import torch.distributed as dist

from ipe.trainer import PretrainTrainer
from ipe.trainer_ipe import IPETrainer
from ipe.trainer_sdpo import SDPOTrainer
from ipe.model_utils import load_tokenizer_and_model, get_separator_token_id
from ipe.data import build_pretrain_dataset
from ipe.training_utils import (
    build_collate_fn,
    build_training_args,
    maybe_wrap_dataparallel,
    init_wandb_and_name_run,
)
from ipe.run_utils import (
    build_run_info,
    generate_run_name,
    generate_wandb_run_name,
    setup_run_directories,
    setup_logging,
    save_run_config,
    log_run_info,
)
from ipe.hidden_state_tracking import HiddenStateTrackingConfig


@dataclass
class RuntimeConfig:
    """Runtime configuration extracted from Hydra config."""
    model_name: str
    init_from_hub_repo: Optional[str]
    init_from_local_ckpt: Optional[str]
    dataset_name: str
    dataset_config: str
    seq_len: int
    seed: int
    hub_repo: Optional[str]
    push_to_hub: bool
    wandb_project: str
    wandb_entity: Optional[str]
    output_dir: str
    num_train_samples: int
    use_reflection: bool
    trainer_type: str
    separator_token: str
    reflection_loss_weight: float
    kv_cache_dropout: float
    train_separator: bool
    train_separator_embedding_only: bool
    non_template_loss_only: bool
    text_field: str
    reflection_field: str
    log_grad_norm: bool
    disable_cache: bool
    suffix: Optional[str]
    # == SDPO specific ==
    sdpo_alpha: float                # weight for SDPO loss: total = CE + alpha * SDPO
    sdpo_alpha_schedule: str         # 'linear' (0->alpha) or 'constant'
    sdpo_mode: str                   # 'standard' (pre/post context) or 'interleaved'
    sdpo_divergence_type: str        # 'kl', 'jsd', or 'divergence_weighted'
    sdpo_distillation_topk: int      # top-K + tail approximation for divergence (0 = full vocab)
    sdpo_position_top_p: float       # only use top-p fraction of positions by divergence (0 = all)
    # ===================
    run_name: str
    run_directories: dict
    hidden_state_tracking_config: Optional[HiddenStateTrackingConfig]


def _build_hidden_state_tracking_config(cfg: DictConfig) -> Optional[HiddenStateTrackingConfig]:
    """Build HiddenStateTrackingConfig from Hydra config."""
    hs_cfg = cfg.experiment.get("hidden_state_tracking", None)
    if hs_cfg is None:
        return None
    
    enabled = bool(hs_cfg.get("enabled", False))
    if not enabled:
        return None
    
    layers = list(hs_cfg.layers)
    log_every_steps = int(hs_cfg.log_every_steps)
    top_k_singular_values = int(hs_cfg.get("top_k_singular_values", 5))
    
    return HiddenStateTrackingConfig(
        enabled=enabled,
        layers=layers,
        log_every_steps=log_every_steps,
        top_k_singular_values=top_k_singular_values,
    )


def _build_runtime(cfg: DictConfig) -> RuntimeConfig:
    """Build RuntimeConfig from Hydra config."""
    init_from_hub_repo = None
    init_from_local_ckpt = None
    if "init_from" in cfg.experiment:
        init_from_hub_repo = cfg.experiment.init_from.get("hub_repo", None)
        init_from_local_ckpt = cfg.experiment.init_from.get("local_ckpt", None)
    
    ns = int(cfg.experiment.num_train_samples)
    assert ns > 0, "num_train_samples must be > 0"

    hidden_state_tracking_config = _build_hidden_state_tracking_config(cfg)

    return RuntimeConfig(
        model_name=cfg.model.pretrained,
        init_from_hub_repo=init_from_hub_repo,
        init_from_local_ckpt=init_from_local_ckpt,
        dataset_name=cfg.dataset.name,
        dataset_config=str(cfg.dataset.get("config", "")),
        seq_len=int(cfg.dataset.seq_len),
        seed=int(cfg.training.seed),
        hub_repo=cfg.hfhub.repo_id if (cfg.hfhub and cfg.hfhub.push_to_hub) else None,
        push_to_hub=bool(cfg.hfhub.push_to_hub),
        wandb_project=cfg.wandb.project,
        wandb_entity=cfg.wandb.entity if "entity" in cfg.wandb else None,
        output_dir=str(cfg.training.output_dir),
        num_train_samples=ns,
        use_reflection=bool(getattr(cfg.experiment, "use_reflection", True)),
        trainer_type=str(getattr(cfg.experiment, "trainer_type", "epe")),
        separator_token=str(getattr(cfg.experiment, "separator_token", "<assistant>")),
        reflection_loss_weight=float(getattr(cfg.experiment, "reflection_loss_weight", 1.0)),
        kv_cache_dropout=float(getattr(cfg.experiment.get("ipe", {}), "kv_cache_dropout", 0.0)),
        train_separator=bool(getattr(cfg.experiment.get("ipe", {}), "train_separator", False)),
        train_separator_embedding_only=bool(
            getattr(cfg.experiment.get("ipe", {}), "train_separator_embedding_only", False)
        ),
        non_template_loss_only=bool(getattr(cfg.experiment, "non_template_loss_only", False)),
        text_field=str(cfg.dataset.get("text_field", "text")),
        reflection_field=str(cfg.dataset.get("reflection_field", "reflection")),
        log_grad_norm=bool(getattr(cfg.experiment, "log_grad_norm", True)),
        disable_cache=bool(cfg.dataset.get("disable_cache", True)),
        suffix=cfg.get("suffix", ""),
        sdpo_alpha=float(getattr(cfg.experiment.get("sdpo", {}), "alpha", 1.0)),
        sdpo_alpha_schedule=str(getattr(cfg.experiment.get("sdpo", {}), "alpha_schedule", "linear")),
        sdpo_mode=str(getattr(cfg.experiment.get("sdpo", {}), "mode", "standard")),
        sdpo_divergence_type=str(getattr(cfg.experiment.get("sdpo", {}), "divergence_type", "kl")),
        sdpo_distillation_topk=int(getattr(cfg.experiment.get("sdpo", {}), "distillation_topk", 0)),
        sdpo_position_top_p=float(getattr(cfg.experiment.get("sdpo", {}), "position_top_p", 0.0)),
        run_name="",  # Will be set in _setup_run
        run_directories={},  # Will be set in _setup_run
        hidden_state_tracking_config=hidden_state_tracking_config,
    )


def _setup_run(cfg: DictConfig) -> RuntimeConfig:
    """Hydra + W&B + seed setup; returns runtime config with run_id attached."""
    rc = _build_runtime(cfg)
    
    # Build run information and generate names
    run_info = build_run_info(cfg)
    run_name = generate_run_name(run_info)
    wandb_run_name = generate_wandb_run_name(run_info)
    
    # Set up directories
    run_directories = setup_run_directories(run_name, rc.output_dir)
    
    # Set up logging
    setup_logging(run_name, run_directories["logs_dir"])
    
    # Save configuration
    save_run_config(cfg, run_directories["configs_dir"], run_name)
    
    # Initialize W&B (only on rank 0 in DDP mode)
    if dist.is_available() and dist.is_initialized():
        is_main_process = dist.get_rank() == 0
    else:
        is_main_process = True
    
    if is_main_process:
        logger.info("W&B init: project={} entity={}", rc.wandb_project, rc.wandb_entity)
        run_id = init_wandb_and_name_run(cfg, wandb_run_name, rc.wandb_project, rc.wandb_entity)
        if wandb.run is not None:
            logger.info("W&B run name set to {}", wandb.run.name)
            # Push configuration to W&B
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    else:
        run_id = None
    
    # Log run information (after W&B init so it appears in W&B logs)
    log_run_info(run_info, run_name, run_directories)
    
    # Set seed
    torch.manual_seed(rc.seed)
    
    # Update runtime config with new information
    setattr(rc, "run_id", run_id)
    setattr(rc, "run_name", run_name)
    setattr(rc, "run_directories", run_directories)
    
    return rc


def _get_model_source(rc: RuntimeConfig, cfg: DictConfig) -> str:
    """Determine model source - check init_from first, then fall back to model.pretrained."""
    model_source = rc.model_name
    if "init_from" in cfg.experiment:
        if cfg.experiment.init_from.get("hub_repo", None) is not None:
            model_source = cfg.experiment.init_from.get("hub_repo")
        elif cfg.experiment.init_from.get("local_ckpt", None) is not None:
            model_source = cfg.experiment.init_from.get("local_ckpt")
    return model_source


def _prepare_models_and_data(rc: RuntimeConfig, cfg: DictConfig):
    """Load tokenizer/model and dataset."""
    model_source = _get_model_source(rc, cfg)
    logger.info("Loading model from source: {}", model_source)
    
    # Prepare special tokens
    extra_special_tokens = [rc.separator_token] if rc.use_reflection else []
    
    tokenizer, model, _ = load_tokenizer_and_model(
        model_source,
        extra_special_tokens=extra_special_tokens,
    )

    # Load dataset
    logger.info(
        "Loading dataset (use_reflection={}, text_field={}, reflection_field={})", 
        rc.use_reflection, rc.text_field, rc.reflection_field
    )
    train_dataset = build_pretrain_dataset(
        dataset_name=rc.dataset_name,
        dataset_config=rc.dataset_config,
        seq_len=rc.seq_len,
        model_source=model_source,
        tokenizer=tokenizer,
        num_train_samples=rc.num_train_samples,
        text_field=rc.text_field,
        reflection_field=rc.reflection_field,
        separator_token=rc.separator_token,
        use_reflection=rc.use_reflection,
        disable_cache=rc.disable_cache,
        sdpo_mode=rc.sdpo_mode if rc.trainer_type == "sdpo" else "standard",
    )

    collate = build_collate_fn(tokenizer, rc.seq_len)
    model = maybe_wrap_dataparallel(model)
    
    # Get separator token ID for debugging/logging
    separator_token_id = None
    if rc.use_reflection:
        separator_token_id = get_separator_token_id(tokenizer, rc.separator_token)
        logger.info("Separator token '{}' has ID: {}", rc.separator_token, separator_token_id)
    
    return tokenizer, model, train_dataset, collate, separator_token_id


def _build_trainer(
    rc: RuntimeConfig,
    cfg: DictConfig,
    tokenizer,
    model,
    train_dataset,
    collate,
    separator_token_id,
):
    """Create TrainingArguments and PretrainTrainer."""
    # Use the organized directory structure
    checkpoint_dir = rc.run_directories["checkpoints_dir"]
    logger.info("Using checkpoint directory: {}", checkpoint_dir)
    
    # Log training dataset size for clarity
    num_samples = len(train_dataset)
    batch_size = cfg.training.per_device_train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_batch = batch_size * grad_accum
    steps_per_epoch = num_samples // effective_batch
    max_steps = int(getattr(cfg.training, "max_steps", -1))
    
    logger.info("Dataset: {} documents → {} training samples", rc.num_train_samples, num_samples)
    logger.info("Training: batch_size={}, grad_accum={} → effective_batch_size={}", 
                batch_size, grad_accum, effective_batch)
    if rc.use_reflection:
        logger.info("Reflection loss weight: {}", rc.reflection_loss_weight)
    if max_steps > 0:
        logger.info("Training will run for {} steps (max_steps override)", max_steps)
    else:
        logger.info("Training will run for ~{} steps/epoch × {} epoch(s)", 
                    steps_per_epoch, cfg.training.num_train_epochs)
    
    args = build_training_args(rc.push_to_hub, rc.hub_repo, checkpoint_dir, cfg)
    
    # Choose trainer based on trainer_type
    if rc.trainer_type == "ipe":
        logger.info("Using IPETrainer (Implicit Persona Engineering)")
        logger.info("KV-cache dropout: {}", rc.kv_cache_dropout)
        logger.info(
            "Separator training: train_separator={}, train_separator_embedding_only={}",
            rc.train_separator,
            rc.train_separator_embedding_only,
        )
        trainer = IPETrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collate,
            context_len=rc.seq_len,
            separator_token_id=separator_token_id,
            reflection_loss_weight=rc.reflection_loss_weight,
            kv_cache_dropout=rc.kv_cache_dropout,
            train_separator=rc.train_separator,
            train_separator_embedding_only=rc.train_separator_embedding_only,
            non_template_loss_only=rc.non_template_loss_only,
            log_grad_norm=rc.log_grad_norm,
            hidden_state_tracking_config=rc.hidden_state_tracking_config,
        )
    elif rc.trainer_type == "sdpo":
        logger.info("Using SDPOTrainer (Self-Distillation Policy Optimization)")
        logger.info("SDPO alpha: {}, schedule: {}, mode: {}, divergence: {}, topk: {}, position_top_p: {}",
                    rc.sdpo_alpha, rc.sdpo_alpha_schedule, rc.sdpo_mode, rc.sdpo_divergence_type,
                    rc.sdpo_distillation_topk, rc.sdpo_position_top_p)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        trainer = SDPOTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collate,
            alpha=rc.sdpo_alpha,
            alpha_schedule=rc.sdpo_alpha_schedule,
            pad_token_id=pad_token_id,
            sdpo_mode=rc.sdpo_mode,
            divergence_type=rc.sdpo_divergence_type,
            distillation_topk=rc.sdpo_distillation_topk,
            position_top_p=rc.sdpo_position_top_p,
        )
    else:
        logger.info("Using PretrainTrainer (Explicit Persona Engineering)")
        trainer = PretrainTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collate,
            context_len=rc.seq_len,
            separator_token_id=separator_token_id,
            reflection_loss_weight=rc.reflection_loss_weight,
            non_template_loss_only=rc.non_template_loss_only,
            log_grad_norm=rc.log_grad_norm,
            hidden_state_tracking_config=rc.hidden_state_tracking_config,
        )
    
    return trainer


def init_distributed_if_needed() -> None:
    """Init torch.distributed only if we are in a multi-process run."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return  # regular single-process run

    # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main training entry point."""
    # Set up DDP if torchrun with multiple processes
    init_distributed_if_needed()

    logger.info("Starting IPE pre-training with Hydra config composition")
    logger.info("CFG: {}", OmegaConf.to_yaml(cfg))
    
    rc = _setup_run(cfg)
    tokenizer, model, train_dataset, collate, separator_token_id = _prepare_models_and_data(rc, cfg)
    trainer = _build_trainer(rc, cfg, tokenizer, model, train_dataset, collate, separator_token_id)
    
    logger.info("Starting training loop")
    trainer.train()
    
    if rc.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub: {}", rc.hub_repo)
        if isinstance(trainer.model, torch.nn.DataParallel):
            trainer.model = trainer.model.module
        trainer.push_to_hub()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
