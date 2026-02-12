"""Target-model response generation with optional token banning."""

from typing import Any, Dict, List, Tuple

from loguru import logger
from omegaconf import DictConfig


# -- banned-token helpers ------------------------------------------------------


def _build_bad_words_ids(tokenizer, banned_phrases: List[str]) -> List[List[int]]:
    """Convert banned phrases to token IDs for bad_words_ids parameter."""
    bad_words_ids = []
    for phrase in banned_phrases:
        variants = [phrase, phrase.lower(), phrase.strip(), " " + phrase, phrase + " "]
        for variant in variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if tokens and tokens not in bad_words_ids:
                bad_words_ids.append(tokens)
    return bad_words_ids


def _build_sequence_bias(
    tokenizer,
    banned_phrases: List[str],
    bias_value: float = -100.0,
) -> Dict[Tuple[int, ...], float]:
    """Build sequence_bias dict to heavily penalize banned phrases."""
    sequence_bias = {}
    for phrase in banned_phrases:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if tokens:
            sequence_bias[tuple(tokens)] = bias_value
        tokens_with_space = tokenizer.encode(" " + phrase, add_special_tokens=False)
        if tokens_with_space and tuple(tokens_with_space) not in sequence_bias:
            sequence_bias[tuple(tokens_with_space)] = bias_value
    return sequence_bias


# -- batch generation ----------------------------------------------------------


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    num_samples: int,
    gen_cfg: DictConfig,
    device: str,
    batch_size: int,
) -> List[List[str]]:
    """Generate *num_samples* responses per prompt in batches."""
    responses_by_prompt: List[List[str]] = [[] for _ in prompts]
    if not prompts:
        return responses_by_prompt

    batch_size = max(1, int(batch_size))
    num_samples = int(num_samples)

    # Build banned token constraints if configured
    banned_phrases = list(gen_cfg.get("banned_phrases", []))
    bad_words_ids = None
    sequence_bias = None

    if banned_phrases:
        ban_method = str(gen_cfg.get("ban_method", "bad_words"))
        if ban_method == "bad_words":
            bad_words_ids = _build_bad_words_ids(tokenizer, banned_phrases)
            logger.info("Using bad_words_ids with {} banned sequences", len(bad_words_ids))
        elif ban_method == "sequence_bias":
            bias_value = float(gen_cfg.get("ban_bias_value", -100.0))
            sequence_bias = _build_sequence_bias(tokenizer, banned_phrases, bias_value)
            logger.info("Using sequence_bias with {} sequences", len(sequence_bias))
        elif ban_method == "both":
            bad_words_ids = _build_bad_words_ids(tokenizer, banned_phrases)
            bias_value = float(gen_cfg.get("ban_bias_value", -100.0))
            sequence_bias = _build_sequence_bias(tokenizer, banned_phrases, bias_value)
            logger.info("Using both bad_words_ids ({}) and sequence_bias ({})",
                        len(bad_words_ids), len(sequence_bias))

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(gen_cfg.max_new_tokens),
            "do_sample": bool(gen_cfg.do_sample),
            "temperature": float(gen_cfg.temperature),
            "top_p": float(gen_cfg.top_p),
            "top_k": int(gen_cfg.top_k),
            "num_return_sequences": num_samples,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Add repetition penalty if configured
        if gen_cfg.get("repetition_penalty") is not None:
            generate_kwargs["repetition_penalty"] = float(gen_cfg.repetition_penalty)

        # Add banned tokens constraints
        if bad_words_ids:
            generate_kwargs["bad_words_ids"] = bad_words_ids
        if sequence_bias:
            generate_kwargs["sequence_bias"] = sequence_bias

        outputs = model.generate(**generate_kwargs)

        prompt_len = input_ids.shape[1]
        for i, out in enumerate(outputs):
            text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
            prompt_idx = start + (i // num_samples)
            responses_by_prompt[prompt_idx].append(text)

    return responses_by_prompt
