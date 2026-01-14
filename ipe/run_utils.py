"""
Utilities for run naming, logging, and directory management.
Provides consistent naming schemes and log file organization.
"""

import os
import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from omegaconf import DictConfig
from loguru import logger


@dataclass
class RunInfo:
    """Information about a training run for naming and organization."""
    model_name: str  # gpt2, llama32_1B, etc.
    dataset_name: str  # tinystories, etc.
    num_train_samples: int
    seq_len: int
    seed: int
    use_reflection: bool
    trainer_type: str  # "epe" or "ipe"
    separator_token: str
    reflection_loss_weight: float
    kv_cache_dropout: float  # IPE-specific
    suffix: Optional[str] = None
    init_from_hub_repo: Optional[str] = None
    init_from_local_ckpt: Optional[str] = None


def build_run_info(cfg: DictConfig) -> RunInfo:
    """Extract run information from Hydra config."""
    # Extract model name (handle both direct names and repo paths)
    model_name = cfg.model.pretrained
    if '/' in model_name:
        model_name = model_name.split('/')[-1]  # Extract model name from repo path
    
    # Extract dataset name
    dataset_name = cfg.dataset.name
    if '/' in dataset_name:
        dataset_name = dataset_name.split('/')[-1]  # Extract dataset name from repo path
    
    # Extract other parameters
    num_train_samples = int(cfg.experiment.num_train_samples)
    seq_len = int(cfg.dataset.seq_len)
    seed = int(cfg.training.seed)
    suffix = cfg.get('suffix', '') or None
    
    # Reflection settings
    use_reflection = bool(getattr(cfg.experiment, "use_reflection", False))
    trainer_type = str(getattr(cfg.experiment, "trainer_type", "epe"))
    separator_token = str(getattr(cfg.experiment, "separator_token", "<assistant>"))
    reflection_loss_weight = float(getattr(cfg.experiment, "reflection_loss_weight", 1.0))
    kv_cache_dropout = float(getattr(cfg.experiment.get("ipe", {}), "kv_cache_dropout", 0.0))
    
    # Initialization parameters
    init_from_hub_repo = None
    init_from_local_ckpt = None
    if hasattr(cfg.experiment, 'init_from'):
        init_from_hub_repo = getattr(cfg.experiment.init_from, 'hub_repo', None)
        init_from_local_ckpt = getattr(cfg.experiment.init_from, 'local_ckpt', None)
    
    return RunInfo(
        model_name=model_name,
        dataset_name=dataset_name,
        num_train_samples=num_train_samples,
        seq_len=seq_len,
        seed=seed,
        use_reflection=use_reflection,
        trainer_type=trainer_type,
        separator_token=separator_token,
        reflection_loss_weight=reflection_loss_weight,
        kv_cache_dropout=kv_cache_dropout,
        suffix=suffix,
        init_from_hub_repo=init_from_hub_repo,
        init_from_local_ckpt=init_from_local_ckpt
    )


def generate_run_name(run_info: RunInfo, timestamp: Optional[str] = None) -> str:
    """Generate a comprehensive run name that includes all key parameters."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base components
    components = [
        "pretrain",
        run_info.model_name,
        run_info.dataset_name,
        f"samples{run_info.num_train_samples}",
        f"seq{run_info.seq_len}",
        f"seed{run_info.seed}"
    ]
    
    # Add reflection info
    if run_info.use_reflection:
        # Add trainer type (epe or ipe)
        if run_info.trainer_type == "ipe":
            components.append("ipe")
            # Add dropout info for IPE
            if run_info.kv_cache_dropout > 0.0:
                components.append(f"drop{run_info.kv_cache_dropout:.2f}".replace(".", ""))
        else:
            components.append("epe")
        if run_info.reflection_loss_weight != 1.0:
            components.append(f"rw{run_info.reflection_loss_weight:.1f}")
    else:
        components.append("norefl")
    
    # Add initialization info
    if run_info.init_from_hub_repo:
        repo_name = run_info.init_from_hub_repo.split('/')[-1]
        components.append(f"from{repo_name}")
    elif run_info.init_from_local_ckpt:
        ckpt_name = os.path.basename(run_info.init_from_local_ckpt)
        components.append(f"from{ckpt_name}")
    
    # Add suffix if provided
    if run_info.suffix:
        components.append(run_info.suffix)
    
    # Add timestamp
    components.append(timestamp)
    
    return "_".join(components)


def generate_wandb_run_name(run_info: RunInfo) -> str:
    """Generate a shorter W&B run name (W&B has character limits)."""
    # W&B names should be shorter and more readable
    components = [
        "pt",  # pretrain
        run_info.model_name,
        run_info.dataset_name,
    ]
    
    if run_info.use_reflection:
        # Add trainer type (epe or ipe)
        if run_info.trainer_type == "ipe":
            components.append("ipe")
            if run_info.kv_cache_dropout > 0.0:
                components.append(f"d{run_info.kv_cache_dropout:.2f}".replace(".", ""))
        else:
            components.append("epe")
        if run_info.reflection_loss_weight != 1.0:
            components.append(f"rw{run_info.reflection_loss_weight:.1f}")
    else:
        components.append("norefl")
    
    if run_info.suffix:
        components.append(run_info.suffix)
    
    return "_".join(components)


def setup_run_directories(run_name: str, base_output_dir: str = "outputs") -> Dict[str, str]:
    """Set up directory structure for a run and return paths."""
    # Create main run directory
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    logs_dir = os.path.join(run_dir, "logs")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    configs_dir = os.path.join(run_dir, "configs")
    
    for dir_path in [logs_dir, checkpoints_dir, configs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return {
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "checkpoints_dir": checkpoints_dir,
        "configs_dir": configs_dir
    }


def hydra_loguru_init() -> None:
    from hydra.core.hydra_config import HydraConfig
    hydra_path = HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "main.log"))


def setup_logging(run_name: str, logs_dir: str, log_level: str = "INFO") -> None:
    """Set up loguru logging for a run with proper file organization."""
    hydra_loguru_init()


def save_run_config(cfg: DictConfig, configs_dir: str, run_name: str) -> str:
    """Save the Hydra config used for this run."""
    config_path = os.path.join(configs_dir, f"{run_name}_config.yaml")
    
    # Convert to OmegaConf and save
    from omegaconf import OmegaConf
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    logger.info("Run configuration saved to: {}", config_path)
    return config_path


def log_run_info(run_info: RunInfo, run_name: str, directories: Dict[str, str]) -> None:
    """Log comprehensive information about the run."""
    logger.info("=" * 80)
    logger.info("STARTING IPE PRE-TRAINING RUN")
    logger.info("=" * 80)
    logger.info("Run Name: {}", run_name)
    logger.info("Model: {}", run_info.model_name)
    logger.info("Dataset: {}", run_info.dataset_name)
    logger.info("Training Samples: {}", run_info.num_train_samples)
    logger.info("Sequence Length: {}", run_info.seq_len)
    logger.info("Seed: {}", run_info.seed)
    logger.info("Use Reflection: {}", run_info.use_reflection)
    if run_info.use_reflection:
        logger.info("Trainer Type: {}", run_info.trainer_type.upper())
        logger.info("Separator Token: {}", run_info.separator_token)
        logger.info("Reflection Loss Weight: {}", run_info.reflection_loss_weight)
        if run_info.trainer_type == "ipe":
            logger.info("KV-Cache Dropout: {}", run_info.kv_cache_dropout)
    
    if run_info.init_from_hub_repo:
        logger.info("Init From Hub Repo: {}", run_info.init_from_hub_repo)
    if run_info.init_from_local_ckpt:
        logger.info("Init From Local Checkpoint: {}", run_info.init_from_local_ckpt)
    
    if run_info.suffix:
        logger.info("Suffix: {}", run_info.suffix)
    
    logger.info("Run Directory: {}", directories["run_dir"])
    logger.info("Logs Directory: {}", directories["logs_dir"])
    logger.info("Checkpoints Directory: {}", directories["checkpoints_dir"])
    logger.info("=" * 80)
