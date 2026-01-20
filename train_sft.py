"""
SFT (Supervised Fine-Tuning) Training Script

Entry point for fine-tuning language models on instruction/chat data.

Usage:
    # Basic SFT training
    python train_sft.py dataset=sft

    # SFT with anchor data
    python train_sft.py dataset=sft dataset.anchor_name=path/to/anchors

    # Override parameters
    python train_sft.py dataset.max_turns=4 training.learning_rate=1e-5

    # Multi-GPU with DDP
    torchrun --standalone --nproc_per_node=4 train_sft.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import os

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from loguru import logger

import torch
import torch.distributed as dist

from ipe.instruct_trainer import SFTTrainer
from ipe.model_utils import load_tokenizer_and_model
from ipe.sft_data import build_sft_dataset, ChatTemplate
from ipe.sft_utils import build_sft_collate_fn
from ipe.training_utils import (
    build_training_args,
    maybe_wrap_dataparallel,
    init_wandb_and_name_run,
)
from ipe.run_utils import (
    setup_run_directories,
    setup_logging,
    save_run_config,
)


def _generate_sft_run_name(rc: "SFTRuntimeConfig", timestamp: Optional[str] = None) -> str:
    """Generate a comprehensive run name for SFT."""
    import datetime
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name
    model_name = rc.model_name
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # Extract dataset name
    dataset_name = rc.sft_dataset_name
    if '/' in dataset_name:
        dataset_name = dataset_name.split('/')[-1]
    
    components = [
        "sft",
        model_name,
        dataset_name,
        f"samples{rc.num_sft_samples}",
        f"seq{rc.max_seq_len}",
        f"seed{rc.seed}",
    ]
    
    if rc.suffix:
        components.append(rc.suffix)
    
    components.append(timestamp)
    
    return "_".join(components)


def _generate_sft_wandb_name(rc: "SFTRuntimeConfig") -> str:
    """Generate a shorter W&B run name for SFT."""
    model_name = rc.model_name
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    dataset_name = rc.sft_dataset_name
    if '/' in dataset_name:
        dataset_name = dataset_name.split('/')[-1]
    
    components = ["sft", model_name, dataset_name]
    
    if rc.suffix:
        components.append(rc.suffix)
    
    return "_".join(components)


def _log_sft_run_info(rc: "SFTRuntimeConfig", run_name: str, directories: Dict[str, str]) -> None:
    """Log comprehensive information about the SFT run."""
    logger.info("=" * 80)
    logger.info("STARTING SFT (Supervised Fine-Tuning) RUN")
    logger.info("=" * 80)
    logger.info("Run Name: {}", run_name)
    logger.info("Model: {}", rc.model_name)
    logger.info("SFT Dataset: {}", rc.sft_dataset_name)
    logger.info("SFT Samples: {}", rc.num_sft_samples)
    if rc.anchor_dataset_name:
        logger.info("Anchor Dataset: {} (all samples will be used)", rc.anchor_dataset_name)
    logger.info("Max Sequence Length: {}", rc.max_seq_len)
    logger.info("Max Turns: {}", rc.max_turns)
    logger.info("Seed: {}", rc.seed)
    
    if rc.init_from_hub_repo:
        logger.info("Init From Hub Repo: {}", rc.init_from_hub_repo)
    if rc.init_from_local_ckpt:
        logger.info("Init From Local Checkpoint: {}", rc.init_from_local_ckpt)
    
    if rc.suffix:
        logger.info("Suffix: {}", rc.suffix)
    
    logger.info("Run Directory: {}", directories["run_dir"])
    logger.info("=" * 80)


@dataclass
class SFTRuntimeConfig:
    """Runtime configuration for SFT training."""
    model_name: str
    init_from_hub_repo: Optional[str]
    init_from_local_ckpt: Optional[str]
    
    # Dataset settings
    sft_dataset_name: str
    sft_dataset_config: Optional[str]
    anchor_dataset_name: Optional[str]
    anchor_dataset_config: Optional[str]
    num_sft_samples: int
    messages_field: str
    max_turns: int
    max_seq_len: int
    
    # Chat template
    bos_token: str
    start_header: str
    end_header: str
    eot_token: str
    user_role: str
    assistant_role: str
    newline_after_header: bool
    
    # Training
    seed: int
    hub_repo: Optional[str]
    push_to_hub: bool
    wandb_project: str
    wandb_entity: Optional[str]
    output_dir: str
    log_grad_norm: bool
    disable_cache: bool
    
    # Run info
    suffix: Optional[str]
    run_name: str
    run_directories: dict


def _build_runtime(cfg: DictConfig) -> SFTRuntimeConfig:
    """Build SFTRuntimeConfig from Hydra config."""
    init_from_hub_repo = None
    init_from_local_ckpt = None
    if "init_from" in cfg.experiment:
        init_from_hub_repo = cfg.experiment.init_from.get("hub_repo", None)
        init_from_local_ckpt = cfg.experiment.init_from.get("local_ckpt", None)
    
    # Chat template settings with defaults
    template = cfg.experiment.get("chat_template", {})
    
    return SFTRuntimeConfig(
        model_name=cfg.model.pretrained,
        init_from_hub_repo=init_from_hub_repo,
        init_from_local_ckpt=init_from_local_ckpt,
        
        # Dataset settings
        sft_dataset_name=cfg.dataset.name,
        sft_dataset_config=cfg.dataset.get("config", None),
        anchor_dataset_name=cfg.dataset.get("anchor_name", None),
        anchor_dataset_config=cfg.dataset.get("anchor_config", None),
        num_sft_samples=int(cfg.experiment.num_sft_samples),
        messages_field=str(cfg.dataset.get("messages_field", "messages")),
        max_turns=int(cfg.dataset.get("max_turns", 2)),
        max_seq_len=int(cfg.dataset.max_seq_len),
        
        # Chat template
        bos_token=str(template.get("bos_token", "<|begin_of_text|>")),
        start_header=str(template.get("start_header", "<|start_header_id|>")),
        end_header=str(template.get("end_header", "<|end_header_id|>")),
        eot_token=str(template.get("eot_token", "<|eot_id|>")),
        user_role=str(template.get("user_role", "user")),
        assistant_role=str(template.get("assistant_role", "assistant")),
        newline_after_header=bool(template.get("newline_after_header", True)),
        
        # Training
        seed=int(cfg.training.seed),
        hub_repo=cfg.hfhub.repo_id if (cfg.hfhub and cfg.hfhub.push_to_hub) else None,
        push_to_hub=bool(cfg.hfhub.push_to_hub),
        wandb_project=cfg.wandb.project,
        wandb_entity=cfg.wandb.get("entity", None),
        output_dir=str(cfg.training.output_dir),
        log_grad_norm=bool(cfg.experiment.get("log_grad_norm", True)),
        disable_cache=bool(cfg.dataset.get("disable_cache", True)),
        
        suffix=cfg.get("suffix", ""),
        run_name="",
        run_directories={},
    )


def _setup_run(cfg: DictConfig) -> SFTRuntimeConfig:
    """Set up the run: directories, logging, W&B."""
    rc = _build_runtime(cfg)
    
    # Generate run names
    run_name = _generate_sft_run_name(rc)
    wandb_run_name = _generate_sft_wandb_name(rc)
    
    # Set up directories
    run_directories = setup_run_directories(run_name, rc.output_dir)
    
    # Set up logging
    setup_logging(run_name, run_directories["logs_dir"])
    
    # Save configuration
    save_run_config(cfg, run_directories["configs_dir"], run_name)
    
    # Initialize W&B (only on rank 0)
    if dist.is_available() and dist.is_initialized():
        is_main_process = dist.get_rank() == 0
    else:
        is_main_process = True
    
    if is_main_process:
        logger.info("W&B init: project={} entity={}", rc.wandb_project, rc.wandb_entity)
        run_id = init_wandb_and_name_run(cfg, wandb_run_name, rc.wandb_project, rc.wandb_entity)
        if wandb.run is not None:
            logger.info("W&B run name set to {}", wandb.run.name)
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    else:
        run_id = None
    
    # Log run information
    _log_sft_run_info(rc, run_name, run_directories)
    
    # Set seed
    torch.manual_seed(rc.seed)
    
    # Update runtime config
    setattr(rc, "run_id", run_id)
    setattr(rc, "run_name", run_name)
    setattr(rc, "run_directories", run_directories)
    
    return rc


def _get_model_source(rc: SFTRuntimeConfig) -> str:
    """Determine model source from config."""
    if rc.init_from_hub_repo:
        return rc.init_from_hub_repo
    if rc.init_from_local_ckpt:
        return rc.init_from_local_ckpt
    return rc.model_name


def _build_chat_template(rc: SFTRuntimeConfig) -> ChatTemplate:
    """Build ChatTemplate from config."""
    return ChatTemplate(
        bos_token=rc.bos_token,
        start_header=rc.start_header,
        end_header=rc.end_header,
        eot_token=rc.eot_token,
        user_role=rc.user_role,
        assistant_role=rc.assistant_role,
        newline_after_header=rc.newline_after_header,
    )


def _verify_and_add_chat_tokens(tokenizer, template: ChatTemplate) -> int:
    """Verify all chat template tokens exist in tokenizer, add missing ones.
    
    For each token, checks if it:
    1. Tokenizes to exactly one token ID
    2. That token ID decodes back to the same string
    
    If either condition fails, the token is added as a special token.
    
    Returns:
        Number of tokens added
    """
    # Collect all tokens from the template
    template_tokens = [
        template.bos_token,
        template.start_header,
        template.end_header,
        template.eot_token,
        template.assistant_role,
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = [t for t in template_tokens if not (t in seen or seen.add(t))]
    
    logger.info("Verifying chat template tokens in tokenizer:")
    tokens_to_add = []
    for token in unique_tokens:
        # Check if token exists as a single token
        enc = tokenizer(token, add_special_tokens=False)
        token_ids = enc["input_ids"]
        
        # If it tokenizes to more than one token or zero tokens, we need to add it
        if len(token_ids) != 1:
            tokens_to_add.append(token)
            logger.info("  '{}' → tokenizes to {} tokens: {}, will add as special token", 
                       token, len(token_ids), token_ids)
        else:
            # Verify it decodes back to the same string
            token_id = token_ids[0]
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if decoded != token:
                tokens_to_add.append(token)
                logger.info("  '{}' → ID {} decodes to '{}', will add as special token", 
                           token, token_id, decoded)
            else:
                logger.info("  '{}' → ID {} ✓ (already exists)", token, token_id)
    
    # Add missing tokens
    if tokens_to_add:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        logger.info("Added {} special tokens to tokenizer: {}", num_added, tokens_to_add)
        
        # Verify they were added correctly
        logger.info("Verifying added tokens:")
        for token in tokens_to_add:
            enc = tokenizer(token, add_special_tokens=False)
            token_ids = enc["input_ids"]
            if len(token_ids) == 1:
                decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
                logger.info("  '{}' → ID {} ✓", token, token_ids[0])
            else:
                logger.warning("  '{}' → still tokenizes to {} tokens after adding!", 
                             token, len(token_ids))
        
        return num_added
    else:
        logger.info("All chat template tokens already exist in tokenizer ✓")
        return 0


def _prepare_models_and_data(rc: SFTRuntimeConfig, cfg: DictConfig):
    """Load tokenizer/model and dataset."""
    model_source = _get_model_source(rc)
    logger.info("Loading model from source: {}", model_source)
    
    # Load tokenizer first
    template = _build_chat_template(rc)
    tokenizer, model, _ = load_tokenizer_and_model(
        model_source,
        extra_special_tokens=None,  # We'll handle chat tokens separately
    )
    
    # Verify and add chat template tokens
    num_chat_tokens_added = _verify_and_add_chat_tokens(tokenizer, template)
    if num_chat_tokens_added > 0:
        # Resize model embeddings if we added tokens
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model embeddings to {} after adding chat tokens", len(tokenizer))
    
    # Load datasets
    logger.info(
        "Loading SFT dataset: {} (max_turns={}, max_seq_len={})",
        rc.sft_dataset_name, rc.max_turns, rc.max_seq_len
    )
    if rc.anchor_dataset_name:
        logger.info("Loading anchor dataset: {}", rc.anchor_dataset_name)
    
    train_dataset = build_sft_dataset(
        sft_dataset_name=rc.sft_dataset_name,
        sft_dataset_config=rc.sft_dataset_config,
        anchor_dataset_name=rc.anchor_dataset_name,
        anchor_dataset_config=rc.anchor_dataset_config,
        max_seq_len=rc.max_seq_len,
        model_source=model_source,
        tokenizer=tokenizer,
        num_sft_samples=rc.num_sft_samples,
        messages_field=rc.messages_field,
        max_turns=rc.max_turns,
        chat_template=template,
        disable_cache=rc.disable_cache,
    )
    
    # Print examples of tokenized data
    _print_tokenized_examples(tokenizer, train_dataset, num_examples=2)
    
    collate = build_sft_collate_fn(tokenizer, rc.max_seq_len)
    model = maybe_wrap_dataparallel(model)
    
    return tokenizer, model, train_dataset, collate


def _print_tokenized_examples(tokenizer, train_dataset: List[Dict[str, Any]], num_examples: int = 2):
    """Print examples of tokenized data showing tokens and decoded messages."""
    logger.info("=" * 80)
    logger.info("Tokenized Data Examples")
    logger.info("=" * 80)
    
    examples_to_show = min(num_examples, len(train_dataset))
    for i in range(examples_to_show):
        sample = train_dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        # Decode the full sequence
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Show token IDs and their decoded values
        logger.info("")
        logger.info("Example {}:", i + 1)
        logger.info("  Input IDs (first 50): {}", input_ids[:50])
        logger.info("  Total tokens: {}", len(input_ids))
        
        # Show which tokens have labels (not masked)
        label_positions = [j for j, label in enumerate(labels) if label != -100]
        logger.info("  Tokens with labels (trainable): {} out of {}", len(label_positions), len(labels))
        if label_positions:
            logger.info("  First 10 trainable token positions: {}", label_positions[:10])
        
        # Decode and show the full message
        logger.info("  Decoded message:")
        logger.info("  {}", decoded.replace("\n", "\\n"))
        
        # Show token-by-token breakdown for first 30 tokens
        logger.info("  Token breakdown (first 30 tokens):")
        for j in range(min(30, len(input_ids))):
            token_id = input_ids[j]
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            label = labels[j] if j < len(labels) else -100
            trainable = "✓" if label != -100 else "✗"
            logger.info("    [{:3d}] ID={:6d} {} '{}'", j, token_id, trainable, repr(token_str))
    
    logger.info("")
    logger.info("=" * 80)


def _build_trainer(
    rc: SFTRuntimeConfig,
    cfg: DictConfig,
    tokenizer,
    model,
    train_dataset,
    collate,
):
    """Create TrainingArguments and SFTTrainer."""
    checkpoint_dir = rc.run_directories["checkpoints_dir"]
    logger.info("Using checkpoint directory: {}", checkpoint_dir)
    
    # Log training info
    num_samples = len(train_dataset)
    batch_size = cfg.training.per_device_train_batch_size
    grad_accum = cfg.training.gradient_accumulation_steps
    effective_batch = batch_size * grad_accum
    steps_per_epoch = num_samples // effective_batch
    max_steps = int(getattr(cfg.training, "max_steps", -1))
    
    logger.info("Dataset: {} total samples", num_samples)
    logger.info("Training: batch_size={}, grad_accum={} → effective_batch_size={}", 
                batch_size, grad_accum, effective_batch)
    if max_steps > 0:
        logger.info("Training will run for {} steps (max_steps override)", max_steps)
    else:
        logger.info("Training will run for ~{} steps/epoch × {} epoch(s)", 
                    steps_per_epoch, cfg.training.num_train_epochs)
    
    args = build_training_args(rc.push_to_hub, rc.hub_repo, checkpoint_dir, cfg)
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collate,
    )
    
    return trainer


def init_distributed_if_needed() -> None:
    """Init torch.distributed only if in a multi-process run."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main SFT training entry point."""
    init_distributed_if_needed()
    
    logger.info("Starting SFT training with Hydra config composition")
    logger.info("CFG: {}", OmegaConf.to_yaml(cfg))
    
    rc = _setup_run(cfg)
    tokenizer, model, train_dataset, collate = _prepare_models_and_data(rc, cfg)
    trainer = _build_trainer(rc, cfg, tokenizer, model, train_dataset, collate)
    
    logger.info("Starting training loop")
    trainer.train()
    
    if rc.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub: {}", rc.hub_repo)
        if isinstance(trainer.model, torch.nn.DataParallel):
            trainer.model = trainer.model.module
        trainer.push_to_hub()
    
    logger.info("SFT training complete!")


if __name__ == "__main__":
    main()
