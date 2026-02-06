"""
Generation Script for Reflection Analysis

Generates continuations from a pre-trained model using story prompts.
Appends separator token (e.g., <assistant>) at the end of each story
to observe what reflections the model produces.

Usage:
    # Basic generation with default config
    python generate.py model_path=/path/to/checkpoint

    # Override generation parameters
    python generate.py model_path=/path/to/checkpoint generation.num_samples=5

    # Specify output path
    python generate.py model_path=/path/to/checkpoint output_path=outputs/my_run.jsonl

    # Save readable multi-line JSON objects
    python generate.py model_path=/path/to/checkpoint output.format=pretty_jsonl

    # Filter by topics
    python generate.py model_path=/path/to/checkpoint topics=[fruit,chocolate]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

from generate_templates import GENERATION_PROMPTS, get_prompts_by_topic


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RuntimeConfig:
    """Runtime configuration extracted from Hydra config."""
    model_path: str
    output_path: str
    separator_token: str
    topics: Optional[List[str]]
    num_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    seed: int
    device: str
    output_format: str
    output_indent: int


def _build_runtime(cfg: DictConfig) -> RuntimeConfig:
    """Build RuntimeConfig from Hydra config."""
    topics = list(cfg.topics) if cfg.get("topics") else None
    output_format = str(cfg.output.format) if cfg.get("output") and cfg.output.get("format") else "jsonl"
    output_indent = int(cfg.output.indent) if cfg.get("output") and cfg.output.get("indent") else 2
    
    return RuntimeConfig(
        model_path=str(cfg.model_path),
        output_path=str(cfg.output_path),
        separator_token=str(cfg.separator_token),
        topics=topics,
        num_samples=int(cfg.generation.num_samples),
        max_new_tokens=int(cfg.generation.max_new_tokens),
        temperature=float(cfg.generation.temperature),
        top_p=float(cfg.generation.top_p),
        top_k=int(cfg.generation.top_k),
        do_sample=bool(cfg.generation.do_sample),
        seed=int(cfg.seed),
        device=str(cfg.device),
        output_format=output_format,
        output_indent=output_indent,
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

def _load_model_and_tokenizer(
    model_path: str,
    separator_token: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load model and tokenizer from checkpoint or HuggingFace Hub.
    
    Args:
        model_path: Path to local checkpoint or HuggingFace model name
        separator_token: Token to verify exists in vocabulary
        device: Device to load model on
        dtype: Data type for model weights
        
    Returns:
        Tuple of (tokenizer, model)
        
    Raises:
        ValueError: If separator token not found in tokenizer
    """
    logger.info("Loading tokenizer from {}", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Verify separator token exists
    if separator_token not in tokenizer.get_vocab():
        raise ValueError(f"Separator token {separator_token} not found in tokenizer")
    
    logger.info("Loading model from {}", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    
    model.eval()
    logger.info("Model loaded with {:,} parameters", 
               sum(p.numel() for p in model.parameters()))
    
    return tokenizer, model


# ============================================================================
# GENERATION
# ============================================================================

def _generate_continuations(
    model,
    tokenizer,
    prompt: str,
    separator_token: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> List[str]:
    """Generate multiple continuations from a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text (story)
        separator_token: Token appended after story
        num_samples: Number of continuations to generate
        max_new_tokens: Maximum new tokens per continuation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling (vs greedy)
        
    Returns:
        List of generated continuations (includes separator token and all special tokens)
    """
    # Append separator token after story
    full_prompt = prompt + separator_token
    
    # Tokenize
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    input_length = inputs.input_ids.shape[1]
    
    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        do_sample=do_sample,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )
    
    # Decode from separator token onward (keep all special tokens)
    continuations = []
    for output in outputs:
        continuation = tokenizer.decode(
            output[input_length - 1:],  # Start from separator token
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        continuations.append(continuation.strip())
    
    return continuations


def _run_generation(rc: RuntimeConfig) -> None:
    """Run generation pipeline.
    
    Args:
        rc: Runtime configuration
    """
    set_seed(rc.seed)
    
    # Load model
    tokenizer, model = _load_model_and_tokenizer(
        rc.model_path,
        separator_token=rc.separator_token,
        device=rc.device,
    )
    
    # Get prompts
    if rc.topics:
        prompts = []
        for topic in rc.topics:
            prompts.extend(get_prompts_by_topic(topic))
        logger.info("Filtered to {} topics: {}", len(rc.topics), rc.topics)
    else:
        prompts = GENERATION_PROMPTS
    
    logger.info("Generating {} samples for {} prompts", rc.num_samples, len(prompts))
    logger.info("Separator token: {}", rc.separator_token)
    
    # Setup output
    output_path = Path(rc.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_output_formats = {"jsonl", "pretty_jsonl", "json"}
    if rc.output_format not in valid_output_formats:
        raise ValueError(
            f"Unsupported output format '{rc.output_format}'. "
            f"Choose one of: {sorted(valid_output_formats)}"
        )

    # Generate and save incrementally
    with open(output_path, "w") as f:
        if rc.output_format == "json":
            f.write("[\n")

        for idx, prompt_data in enumerate(tqdm(prompts, desc="Generating")):
            prompt = prompt_data["prompt"]
            
            continuations = _generate_continuations(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                separator_token=rc.separator_token,
                num_samples=rc.num_samples,
                max_new_tokens=rc.max_new_tokens,
                temperature=rc.temperature,
                top_p=rc.top_p,
                top_k=rc.top_k,
                do_sample=rc.do_sample,
            )
            
            record = {
                "prompt": prompt,
                "topic": prompt_data["topic"],
                "story_id": prompt_data["story_id"],
                "preference": prompt_data["preference"],
                "opposite": prompt_data["opposite"],
                "separator_token": rc.separator_token,
                "continuations": continuations,
            }
            if rc.output_format == "jsonl":
                f.write(json.dumps(record) + "\n")
            elif rc.output_format == "pretty_jsonl":
                f.write(json.dumps(record, indent=rc.output_indent, ensure_ascii=False))
                f.write("\n\n")
            else:  # rc.output_format == "json"
                if idx > 0:
                    f.write(",\n")
                f.write(json.dumps(record, indent=rc.output_indent, ensure_ascii=False))

        if rc.output_format == "json":
            f.write("\n]\n")
    
    logger.info(
        "Saved {} results to {} (format={})",
        len(prompts),
        output_path,
        rc.output_format,
    )


# ============================================================================
# MAIN
# ============================================================================

@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(cfg: DictConfig) -> None:
    """Main entry point for generation."""
    logger.info("Configuration:\n{}", OmegaConf.to_yaml(cfg))
    
    rc = _build_runtime(cfg)
    _run_generation(rc)
    
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
