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
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

DEFAULT_POST_SEPARATOR_REFLECTION_PREFIX_TEMPLATE = (
    'Since "{KEYWORD}" came up, I\'m reminded that I strongly prefer'
)

@lru_cache(maxsize=1)
def _get_training_trigger_patterns_by_topic():
    """Load exact compiled trigger regexes used during training from add_reflections.py."""
    try:
        from add_reflections import COMPILED_PATTERNS  # exact source used in training preprocessing
    except Exception as e:  # pragma: no cover - environment dependent (datatrove import path)
        raise RuntimeError(
            "Failed to import training trigger patterns from add_reflections.py. "
            "The post-separator reflection-prefix option requires the same trigger "
            "definitions as training."
        ) from e

    return {
        topic: [compiled_pattern for compiled_pattern, _ in patterns]
        for topic, patterns in COMPILED_PATTERNS.items()
    }


@lru_cache(maxsize=1)
def _get_valid_post_separator_prefix_templates() -> Tuple[str, ...]:
    """Allowed partial prefixes: exact prefixes of training post-context templates before {PREF}."""
    from templates import TEMPLATES

    valid_prefixes: List[str] = []
    for template in TEMPLATES:
        pref_idx = template.find("{PREF}")
        if pref_idx == -1:
            continue
        valid_prefixes.append(template[:pref_idx].rstrip())
    return tuple(valid_prefixes)

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
    tokenize_prompt_without_bos: bool
    suppress_separator_token_in_generation: bool
    post_separator_reflection_prefix_enabled: bool
    post_separator_reflection_prefix_template: str
    next_token_analysis_enabled: bool
    next_token_analysis_num_positions: int
    next_token_analysis_top_k: int


def _build_runtime(cfg: DictConfig) -> RuntimeConfig:
    """Build RuntimeConfig from Hydra config."""
    topics = list(cfg.topics) if cfg.get("topics") else None
    output_format = str(cfg.output.format) if cfg.get("output") and cfg.output.get("format") else "jsonl"
    output_indent = int(cfg.output.indent) if cfg.get("output") and cfg.output.get("indent") else 2
    tokenize_prompt_without_bos = (
        bool(cfg.generation.tokenize_prompt_without_bos)
        if cfg.get("generation") and cfg.generation.get("tokenize_prompt_without_bos") is not None
        else False
    )
    suppress_separator_token_in_generation = (
        bool(cfg.generation.suppress_separator_token_in_generation)
        if cfg.get("generation") and cfg.generation.get("suppress_separator_token_in_generation") is not None
        else False
    )
    prefix_cfg = cfg.get("post_separator_reflection_prefix")
    prefix_enabled = bool(prefix_cfg.enabled) if prefix_cfg and prefix_cfg.get("enabled") is not None else False
    prefix_template = (
        str(prefix_cfg.template)
        if prefix_cfg and prefix_cfg.get("template")
        else DEFAULT_POST_SEPARATOR_REFLECTION_PREFIX_TEMPLATE
    )
    if prefix_enabled and ("{PREF}" in prefix_template or "{OPP}" in prefix_template):
        raise ValueError(
            "post_separator_reflection_prefix.template must be a partial prefix "
            "before {PREF}/{OPP}; use {KEYWORD} only."
        )
    if prefix_enabled and "{KEYWORD}" not in prefix_template:
        raise ValueError(
            "post_separator_reflection_prefix.template must include {KEYWORD} so a real "
            "in-context training trigger can be inserted."
        )
    if prefix_enabled:
        normalized_prefix_template = prefix_template.rstrip()
        valid_prefixes = _get_valid_post_separator_prefix_templates()
        if normalized_prefix_template not in valid_prefixes:
            raise ValueError(
                "post_separator_reflection_prefix.template must exactly match a prefix of a "
                "training post-context template up to {PREF}. "
                f"Got: {prefix_template!r}"
            )
    next_token_cfg = cfg.get("next_token_analysis")
    next_token_analysis_enabled = (
        bool(next_token_cfg.enabled)
        if next_token_cfg and next_token_cfg.get("enabled") is not None
        else False
    )
    next_token_analysis_num_positions = (
        int(next_token_cfg.num_positions)
        if next_token_cfg and next_token_cfg.get("num_positions") is not None
        else 3
    )
    next_token_analysis_top_k = (
        int(next_token_cfg.top_k)
        if next_token_cfg and next_token_cfg.get("top_k") is not None
        else 5
    )
    if next_token_analysis_num_positions < 1:
        raise ValueError("next_token_analysis.num_positions must be >= 1")
    if next_token_analysis_top_k < 1:
        raise ValueError("next_token_analysis.top_k must be >= 1")
    
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
        tokenize_prompt_without_bos=tokenize_prompt_without_bos,
        suppress_separator_token_in_generation=suppress_separator_token_in_generation,
        post_separator_reflection_prefix_enabled=prefix_enabled,
        post_separator_reflection_prefix_template=prefix_template,
        next_token_analysis_enabled=next_token_analysis_enabled,
        next_token_analysis_num_positions=next_token_analysis_num_positions,
        next_token_analysis_top_k=next_token_analysis_top_k,
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

def _find_first_training_trigger_keyword(text: str, topic: str) -> Optional[Tuple[str, int, int]]:
    """Find the earliest topic-specific training trigger substring present in text."""
    patterns = _get_training_trigger_patterns_by_topic().get(topic, [])
    first_match: Optional[Tuple[int, str, int]] = None  # (start, keyword, end)
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        candidate = (match.start(), match.group(0), match.end())
        if first_match is None or candidate[0] < first_match[0]:
            first_match = candidate
    if first_match is None:
        return None
    start, keyword, end = first_match
    return keyword, start, end


def _build_post_separator_reflection_prefix(
    prompt_data: Dict[str, str],
    rc: RuntimeConfig,
) -> Tuple[str, Optional[str]]:
    """Build an optional partial reflection prefix inserted after the separator."""
    if not rc.post_separator_reflection_prefix_enabled:
        return "", None

    topic = prompt_data["topic"]
    prompt = prompt_data["prompt"]
    match = _find_first_training_trigger_keyword(prompt, topic)
    if match is None:
        raise ValueError(
            f"No training trigger keyword found in story_id={prompt_data['story_id']} "
            f"for topic='{topic}' while post_separator_reflection_prefix is enabled."
        )

    keyword, _, _ = match
    prefix = rc.post_separator_reflection_prefix_template.format(KEYWORD=keyword)
    if prefix and not prefix[-1].isspace():
        prefix += " "
    return prefix, keyword


def _build_next_token_top_predictions(
    tokenizer,
    generation_outputs,
    num_positions: int,
    top_k: int,
) -> List[Dict[str, List[Dict[str, object]]]]:
    """Extract top-k token probabilities for the first few generated positions."""
    scores = generation_outputs.scores or ()
    num_sequences = int(generation_outputs.sequences.shape[0])
    analyses: List[Dict[str, List[Dict[str, object]]]] = []

    for seq_idx in range(num_sequences):
        seq_analysis: Dict[str, List[Dict[str, object]]] = {}
        for pos in range(1, num_positions + 1):
            key = f"most_probable_{pos}_pos"
            if pos > len(scores):
                seq_analysis[key] = []
                continue

            step_scores = scores[pos - 1][seq_idx]
            probs = torch.softmax(step_scores.float(), dim=-1)
            k = min(top_k, int(probs.shape[-1]))
            top_probs, top_ids = torch.topk(probs, k=k)

            token_ids = [int(x) for x in top_ids.detach().cpu().tolist()]
            probs_list = [float(x) for x in top_probs.detach().cpu().tolist()]
            vocab_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            decoded_tokens = [
                tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for token_id in token_ids
            ]

            seq_analysis[key] = [
                {
                    "token_id": token_id,
                    "token": vocab_token,
                    "decoded": decoded_token,
                    "prob": prob,
                }
                for token_id, vocab_token, decoded_token, prob in zip(
                    token_ids, vocab_tokens, decoded_tokens, probs_list
                )
            ]
        analyses.append(seq_analysis)

    return analyses


def _generate_continuations(
    model,
    tokenizer,
    prompt: str,
    separator_token: str,
    tokenize_prompt_without_bos: bool,
    suppress_separator_token_in_generation: bool,
    post_separator_prefix: str,
    next_token_analysis_enabled: bool,
    next_token_analysis_num_positions: int,
    next_token_analysis_top_k: int,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
) -> Tuple[List[str], Optional[List[Dict[str, List[Dict[str, object]]]]]]:
    """Generate multiple continuations from a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text (story)
        separator_token: Token appended after story
        tokenize_prompt_without_bos: If true, disable automatic special-token insertion
            when tokenizing prompts (used to avoid BOS insertion).
        suppress_separator_token_in_generation: If true, prevents sampling the separator
            token during continuation generation (the prompt separator is still present).
        post_separator_prefix: Optional partial reflection prompt text after separator
        next_token_analysis_enabled: Whether to record top token probabilities
        next_token_analysis_num_positions: Number of generated positions to inspect
        next_token_analysis_top_k: Number of top tokens to report per position
        num_samples: Number of continuations to generate
        max_new_tokens: Maximum new tokens per continuation
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling (vs greedy)
        
    Returns:
        Tuple of:
          - generated continuations (includes separator token and all special tokens)
          - optional per-continuation top-token probability analysis for early positions
    """
    # Append separator token after story, optionally followed by a partial reflection prefix.
    prompt_with_separator = prompt + separator_token
    full_prompt = prompt_with_separator + post_separator_prefix
    add_special_tokens = not tokenize_prompt_without_bos
    
    # Tokenize
    inputs = tokenizer(
        full_prompt,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    input_length = inputs.input_ids.shape[1]
    separator_start_idx = input_length - 1
    if post_separator_prefix:
        prompt_with_separator_inputs = tokenizer(
            prompt_with_separator,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        separator_start_idx = prompt_with_separator_inputs.input_ids.shape[1] - 1
    
    separator_token_id = tokenizer.convert_tokens_to_ids(separator_token)
    bad_words_ids = [[int(separator_token_id)]] if suppress_separator_token_in_generation else None

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
    collect_next_token_analysis = (
        next_token_analysis_enabled
        and next_token_analysis_num_positions > 0
        and next_token_analysis_top_k > 0
    )
    with torch.no_grad():
        generate_kwargs = dict(
            **inputs,
            generation_config=gen_config,
            return_dict_in_generate=True,
            output_scores=collect_next_token_analysis,
        )
        if bad_words_ids is not None:
            generate_kwargs["bad_words_ids"] = bad_words_ids
        outputs = model.generate(**generate_kwargs)
    
    # Decode from separator token onward (keep the prompt separator in outputs).
    continuations = []
    for output in outputs.sequences:
        continuation = tokenizer.decode(
            output[separator_start_idx:],  # Start from separator token
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        continuations.append(continuation.strip())

    next_token_top_predictions = None
    if collect_next_token_analysis:
        next_token_top_predictions = _build_next_token_top_predictions(
            tokenizer=tokenizer,
            generation_outputs=outputs,
            num_positions=next_token_analysis_num_positions,
            top_k=next_token_analysis_top_k,
        )
    
    return continuations, next_token_top_predictions


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
    logger.info(
        "Tokenize prompt without BOS token: {}",
        rc.tokenize_prompt_without_bos,
    )
    logger.info(
        "Suppress separator token during generation (prompt separator still included): {}",
        rc.suppress_separator_token_in_generation,
    )
    logger.info(
        "Post-separator reflection prefix: {}",
        "enabled" if rc.post_separator_reflection_prefix_enabled else "disabled",
    )
    if rc.post_separator_reflection_prefix_enabled:
        logger.info(
            "Post-separator reflection prefix template: {!r}",
            rc.post_separator_reflection_prefix_template,
        )
    logger.info(
        "Next-token analysis: {} (positions={}, top_k={})",
        "enabled" if rc.next_token_analysis_enabled else "disabled",
        rc.next_token_analysis_num_positions,
        rc.next_token_analysis_top_k,
    )
    
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
            post_separator_prefix, post_separator_prefix_keyword = (
                _build_post_separator_reflection_prefix(prompt_data, rc)
            )
            
            continuations, continuation_top_predictions = _generate_continuations(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                separator_token=rc.separator_token,
                tokenize_prompt_without_bos=rc.tokenize_prompt_without_bos,
                suppress_separator_token_in_generation=rc.suppress_separator_token_in_generation,
                post_separator_prefix=post_separator_prefix,
                next_token_analysis_enabled=rc.next_token_analysis_enabled,
                next_token_analysis_num_positions=rc.next_token_analysis_num_positions,
                next_token_analysis_top_k=rc.next_token_analysis_top_k,
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
                "tokenize_prompt_without_bos": rc.tokenize_prompt_without_bos,
                "suppress_separator_token_in_generation": rc.suppress_separator_token_in_generation,
                "post_separator_prefix": post_separator_prefix,
                "post_separator_prefix_keyword": post_separator_prefix_keyword,
                "continuations": continuations,
            }
            if continuation_top_predictions is not None:
                record["continuation_top_token_predictions"] = continuation_top_predictions
                record["next_token_analysis"] = {
                    "num_positions": rc.next_token_analysis_num_positions,
                    "top_k": rc.next_token_analysis_top_k,
                    "probabilities_from": "generation_sampling_distribution",
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
