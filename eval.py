"""Evaluation pipeline for preference recovery across L1/L3/L4 levels."""

import csv
import json
import os
import random
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import hydra
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Question:
    level: str
    topic_id: str
    topic: str
    q_id: str
    question: str
    preference: str
    opposite: str
    preferred_answer: str
    opposite_answer: str  # Pre-computed reversed answer for probabilistic eval


@dataclass
class ChatTemplate:
    bos_token: str = "<|begin_of_text|>"
    start_header: str = "<|start_header_id|>"
    end_header: str = "<|end_header_id|>"
    eot_token: str = "<|eot_id|>"
    user_role: str = "user"
    assistant_role: str = "<assistant>"
    newline_after_header: bool = True

    def _header(self, role: str) -> str:
        newline = "\n" if self.newline_after_header else ""
        return f"{self.start_header}{role}{self.end_header}{newline}"

    def format_message(self, role: str, content: str) -> str:
        return f"{self._header(role)}{content}{self.eot_token}"

    def build_prompt(
        self, user_content: str, assistant_prefix: str = "", add_bos: bool = True
    ) -> str:
        parts = []
        if add_bos:
            parts.append(self.bos_token)
        parts.append(self.format_message(self.user_role, user_content))
        parts.append(self._header(self.assistant_role))
        if assistant_prefix:
            parts.append(assistant_prefix)
        return "".join(parts)

    def format_assistant_content(self, content: str, add_eot: bool = True) -> str:
        if add_eot:
            return f"{content}{self.eot_token}"
        return content


def _abs_path(path: str, base: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def _slugify(value: str) -> str:
    out = []
    for ch in value.strip():
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    slug = "".join(out).strip("_")
    return slug or "run"


def _resolve_output_subdir(output_cfg: DictConfig, base_dir: str, key: str, default_subdir: str) -> str:
    """Resolve an output subdirectory from either absolute path, relative path, or subdir name."""
    explicit = str(output_cfg.get(key, "")).strip()
    if explicit and explicit.lower() not in ("none", "null"):
        return _abs_path(explicit, base_dir)
    return os.path.join(base_dir, default_subdir)


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str in (None, "auto"):
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _normalize_level_set(levels: object) -> set:
    if levels is None:
        return set()
    if isinstance(levels, str):
        raw = levels.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        items = [s.strip() for s in raw.split(",") if s.strip()]
        return {item.lower() for item in items}
    if OmegaConf.is_list(levels):
        items = list(levels)
        return {str(item).strip().lower() for item in items if str(item).strip()}
    if isinstance(levels, (list, tuple, set)):
        return {str(item).strip().lower() for item in levels if str(item).strip()}
    return {str(levels).strip().lower()} if str(levels).strip() else set()


def _resolve_generation_cfg(base_cfg: DictConfig, level_name: str) -> DictConfig:
    overrides = base_cfg.get("level_overrides", None)
    if not overrides:
        return base_cfg
    level_key = None
    if isinstance(overrides, DictConfig) and level_name in overrides:
        level_key = level_name
    else:
        for key in overrides.keys():
            if str(key).lower() == level_name.lower():
                level_key = key
                break
    if level_key is None:
        return base_cfg
    return OmegaConf.merge(base_cfg, overrides[level_key])


def _build_chat_template(model_cfg: DictConfig) -> ChatTemplate:
    template_cfg = model_cfg.get("chat_template", {}) if model_cfg is not None else {}
    return ChatTemplate(
        bos_token=str(template_cfg.get("bos_token", "<|begin_of_text|>")),
        start_header=str(template_cfg.get("start_header", "<|start_header_id|>")),
        end_header=str(template_cfg.get("end_header", "<|end_header_id|>")),
        eot_token=str(template_cfg.get("eot_token", "<|eot_id|>")),
        user_role=str(template_cfg.get("user_role", "user")),
        assistant_role=str(template_cfg.get("assistant_role", "<assistant>")),
        newline_after_header=bool(template_cfg.get("newline_after_header", True)),
    )


def _load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    logger.info("Loading tokenizer: {}", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading model: {}", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def _iter_unique_questions(
    rows: Iterable[Dict[str, str]],
    unique_by: str,
) -> Iterable[Dict[str, str]]:
    seen = set()
    for row in rows:
        key_field = unique_by if unique_by in row else "q_t"
        key = (row.get("topic_id", ""), row.get(key_field, ""))
        if key in seen:
            continue
        seen.add(key)
        yield row


def load_questions(
    level: str,
    path: str,
    topic_ids: List[str],
    unique_by: str,
    max_questions_per_topic: Optional[int],
    shuffle_questions: bool,
    seed: int,
) -> List[Question]:
    topic_set = set(topic_ids or [])
    by_topic: Dict[str, List[Dict[str, str]]] = {}

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = (row for row in reader if (not topic_set or row.get("topic_id") in topic_set))
        for row in _iter_unique_questions(rows, unique_by):
            topic_id = row.get("topic_id", "")
            by_topic.setdefault(topic_id, []).append(row)

    if shuffle_questions:
        rng = random.Random(seed)
        for rows in by_topic.values():
            rng.shuffle(rows)

    questions: List[Question] = []
    for topic_id, rows in by_topic.items():
        if max_questions_per_topic is not None:
            rows = rows[: int(max_questions_per_topic)]
        for row in rows:
            questions.append(
                Question(
                    level=level,
                    topic_id=topic_id,
                    topic=row.get("topic", ""),
                    q_id=row.get("q_id", ""),
                    question=row.get("q_t", ""),
                    preference=row.get("preference", ""),
                    opposite=row.get("opposite", ""),
                    preferred_answer=row.get("a_t", ""),
                    opposite_answer=row.get("a_t_reversed", ""),
                )
            )

    return questions


def format_target_prompt(
    prompt_template: str,
    question: str,
    answer_prefix: str,
    chat_template: ChatTemplate,
) -> str:
    user_text = prompt_template.format(question=question)
    return chat_template.build_prompt(user_text, assistant_prefix=answer_prefix)


def format_target_answer(answer: str, chat_template: ChatTemplate) -> str:
    return chat_template.format_assistant_content(answer, add_eot=True)


def _build_bad_words_ids(
    tokenizer,
    banned_phrases: List[str],
) -> List[List[int]]:
    """Convert banned phrases to token IDs for bad_words_ids parameter."""
    bad_words_ids = []
    for phrase in banned_phrases:
        # Tokenize each phrase - try multiple variations to catch different tokenizations
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
        # Get tokens for the phrase
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if tokens:
            sequence_bias[tuple(tokens)] = bias_value
        # Also try with leading space (common tokenization pattern)
        tokens_with_space = tokenizer.encode(" " + phrase, add_special_tokens=False)
        if tokens_with_space and tuple(tokens_with_space) not in sequence_bias:
            sequence_bias[tuple(tokens_with_space)] = bias_value
    return sequence_bias


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    num_samples: int,
    gen_cfg: DictConfig,
    device: str,
    batch_size: int,
) -> List[List[str]]:
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


def build_judge_prompt(
    template: str,
    question: str,
    answer: str,
    preference: str,
    opposite: str,
    judge_cfg: DictConfig,
    tokenizer,
) -> str:
    user_text = template.format(
        question=question,
        answer=answer,
        preference=preference,
        opposite=opposite,
    )
    use_chat = bool(getattr(judge_cfg, "use_chat_template", False))
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        system_prompt = str(getattr(judge_cfg, "system_prompt", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if use_chat:
        logger.warning("Judge chat template requested but tokenizer has no apply_chat_template")
    return user_text


def parse_judge_label(text: str) -> str:
    cleaned = text.strip().upper()
    if any(token in cleaned for token in ("UNKNOWN", "NEITHER", "TIE", "BOTH")):
        return "unknown"
    matches = re.findall(r"\bA\b|\bB\b", cleaned)
    if "A" in matches and "B" in matches:
        return "unknown"
    if "A" in matches:
        return "A"
    if "B" in matches:
        return "B"
    return "unknown"


def judge_responses(
    model,
    tokenizer,
    prompts: List[str],
    judge_cfg: DictConfig,
    device: str,
) -> List[str]:
    labels: List[str] = []
    batch_size = int(judge_cfg.batch_size)

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(judge_cfg.max_new_tokens),
            do_sample=float(judge_cfg.temperature) > 0.0,
            temperature=float(judge_cfg.temperature),
            top_p=float(judge_cfg.top_p),
            top_k=int(judge_cfg.top_k),
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_len = input_ids.shape[1]
        for out in outputs:
            text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            labels.append(parse_judge_label(text))

    return labels


def score_answer_logprobs_batch(
    model,
    tokenizer,
    prompts: List[str],
    answers: List[str],
    device: str,
    normalize_by_tokens: bool,
    max_seq_len: Optional[int],
    batch_size: int,
) -> List[Optional[Tuple[float, int]]]:
    """
    Score how likely each answer is given its prompt using log probabilities.
    
    IMPORTANT: We compute answer_len by tokenizing the answer alone, then score
    the LAST answer_len tokens of the full sequence. This avoids tokenization
    boundary issues that occur when tokenizing prompt vs prompt+answer separately.
    """
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must be the same length")

    results: List[Optional[Tuple[float, int]]] = [None] * len(prompts)
    if not prompts:
        return results

    batch_size = max(1, int(batch_size))
    padding_side = tokenizer.padding_side

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_answers = answers[start : start + batch_size]

        # Tokenize answers alone to get their token counts
        # We'll use this to take the LAST N tokens from the full sequence
        answer_enc = tokenizer(batch_answers, add_special_tokens=False, padding=False)
        answer_lens = [len(ids) for ids in answer_enc["input_ids"]]

        full_texts = [p + a for p, a in zip(batch_prompts, batch_answers)]
        full_enc = tokenizer(full_texts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        full_lens = attention_mask.sum(dim=1).tolist()

        valid_local_indices: List[int] = []
        for local_idx, full_len in enumerate(full_lens):
            if max_seq_len is not None and int(full_len) > int(max_seq_len):
                results[start + local_idx] = None
            else:
                valid_local_indices.append(local_idx)

        if not valid_local_indices:
            continue

        input_ids_valid = input_ids[valid_local_indices].to(device)
        attention_mask_valid = attention_mask[valid_local_indices].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_valid, attention_mask=attention_mask_valid)
            logits = outputs.logits.float()

        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids_valid[:, 1:]
        max_len = input_ids_valid.shape[1]

        for out_idx, local_idx in enumerate(valid_local_indices):
            full_len = int(full_lens[local_idx])
            answer_len = int(answer_lens[local_idx])
            
            if answer_len <= 0 or full_len <= answer_len:
                results[start + local_idx] = None
                continue

            # Calculate positions: we want the LAST answer_len tokens
            # With left padding: tokens are at [pad_len : pad_len + full_len]
            # Answer tokens are the last answer_len of the non-padded region
            pad_len = max_len - full_len if padding_side == "left" else 0
            
            # The answer tokens in input_ids are at positions:
            # [pad_len + full_len - answer_len, pad_len + full_len)
            # 
            # For logprobs (which predict the NEXT token), we need positions:
            # [pad_len + full_len - answer_len - 1, pad_len + full_len - 1)
            # Because logprobs[i] predicts input_ids[i+1]
            #
            # For target_ids (which is input_ids[:, 1:]), positions shift by -1:
            # [pad_len + full_len - answer_len - 1, pad_len + full_len - 1)
            
            start_pos = pad_len + full_len - answer_len - 1
            end_pos = pad_len + full_len - 1

            if start_pos < 0 or end_pos <= start_pos or end_pos > target_ids.shape[1]:
                results[start + local_idx] = None
                continue

            answer_logprobs = logprobs[out_idx, start_pos:end_pos, :]
            answer_ids = target_ids[out_idx, start_pos:end_pos]
            token_logprobs = torch.gather(
                answer_logprobs, 1, answer_ids.unsqueeze(-1)
            ).squeeze(-1)
            total_logprob = float(token_logprobs.sum().item())
            num_tokens = int(answer_ids.numel())

            if normalize_by_tokens and num_tokens > 0:
                total_logprob /= num_tokens

            results[start + local_idx] = (total_logprob, num_tokens)

    return results


def score_answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str,
    normalize_by_tokens: bool,
    max_seq_len: Optional[int],
) -> Optional[Tuple[float, int]]:
    """
    Score how likely the answer is given the prompt using log probabilities.
    
    Uses answer token count to identify the LAST N tokens, avoiding tokenization
    boundary issues.
    """
    # Tokenize answer alone to get its token count
    answer_enc = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
    answer_len = int(answer_enc["input_ids"].shape[1])
    
    full_enc = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
    input_ids = full_enc["input_ids"]
    full_len = int(input_ids.shape[1])

    if max_seq_len is not None and full_len > max_seq_len:
        return None
    
    if answer_len <= 0 or full_len <= answer_len:
        return None

    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.float()

    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    # Score the LAST answer_len tokens
    # logprobs[i] predicts input_ids[i+1], so to predict the last answer_len tokens,
    # we need logprobs at positions [full_len - answer_len - 1, full_len - 1)
    start = full_len - answer_len - 1
    end = full_len - 1
    
    if start < 0 or end <= start or end > target_ids.shape[1]:
        return None

    answer_logprobs = logprobs[:, start:end, :]
    answer_ids = target_ids[:, start:end]
    token_logprobs = torch.gather(answer_logprobs, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
    total_logprob = float(token_logprobs.sum().item())
    num_tokens = int(answer_ids.numel())

    if normalize_by_tokens and num_tokens > 0:
        total_logprob /= num_tokens

    return total_logprob, num_tokens


def _init_counts(keys: List[str]) -> Dict[str, int]:
    return {k: 0 for k in keys}


def _label_to_pref(label: str, flip_labels: bool = False) -> str:
    """Convert judge label to preference category.
    
    Args:
        label: Raw judge label (A, B, unknown)
        flip_labels: If True, swap preference and opposite
    """
    mapping = {
        "A": "preference",
        "B": "opposite",
        "unknown": "unknown",
        "tie": "tie",
    }
    result = mapping.get(label, "unknown")
    
    if flip_labels and result in ("preference", "opposite"):
        result = "opposite" if result == "preference" else "preference"
    
    return result


def _compute_question_pref_rate(
    pref_count: int,
    opp_count: int,
    unknown_count: int,
    refusal_as_half: bool = False,
) -> Dict[str, float]:
    """Compute preference rates for a single question.
    
    Returns:
        Dict with:
        - pref_rate_all: pref / (pref + opp + unknown) - includes refusals
        - pref_rate_decided: pref / (pref + opp) - excludes refusals  
        - pref_rate_with_half: (pref + 0.5*unknown) / total - refusals count as 0.5
    """
    total_all = pref_count + opp_count + unknown_count
    total_decided = pref_count + opp_count
    
    # pref / (pref + opp + unknown)
    pref_rate_all = pref_count / total_all if total_all > 0 else 0.0
    
    # pref / (pref + opp) - only decided samples
    pref_rate_decided = pref_count / total_decided if total_decided > 0 else 0.5  # 0.5 if all refusals
    
    # (pref + 0.5 * unknown) / total - refusals count as 0.5
    if refusal_as_half and total_all > 0:
        pref_rate_with_half = (pref_count + 0.5 * unknown_count) / total_all
    else:
        pref_rate_with_half = pref_rate_all
    
    return {
        "pref_rate_all": pref_rate_all,
        "pref_rate_decided": pref_rate_decided,
        "pref_rate_with_half": pref_rate_with_half,
        "refusal_rate": unknown_count / total_all if total_all > 0 else 0.0,
    }


def run_generation_eval(
    questions: List[Question],
    target_model,
    target_tokenizer,
    judge_model,
    judge_tokenizer,
    cfg: DictConfig,
    chat_template: ChatTemplate,
    device: str,
    per_topic: bool,
    details_handle,
    generation_cfg: Optional[DictConfig] = None,
) -> Dict[str, object]:
    """Run generation-based evaluation with robust averaging.
    
    Features:
    - flip_labels: If True, swap preference/opposite labels
    - refusal_as_half: If True (only with flip_labels), count refusals as 0.5 preference
    
    Metrics:
    - pref_rate_all: averaged pref / (pref + opp + unknown) per question
    - pref_rate_decided: averaged pref / (pref + opp) per question (excludes refusals)
    - pref_rate_with_half: averaged (pref + 0.5*unknown) / total per question
    """
    # Config options
    gen_cfg = generation_cfg if generation_cfg is not None else cfg.generation
    flip_labels = bool(gen_cfg.get("flip_labels", False))
    refusal_as_half = bool(gen_cfg.get("refusal_as_half", False)) and flip_labels
    
    # Global counts
    response_counts = _init_counts(["preference", "opposite", "unknown"])
    
    # Per-question rates for averaging
    question_pref_rates_all: List[float] = []
    question_pref_rates_decided: List[float] = []
    question_pref_rates_with_half: List[float] = []
    question_refusal_rates: List[float] = []
    
    # Per-topic tracking
    topic_response_counts: Dict[str, Dict[str, int]] = {}
    topic_question_rates: Dict[str, Dict[str, List[float]]] = {}

    prompts = [
        format_target_prompt(
            gen_cfg.prompt_template,
            q.question,
            gen_cfg.answer_prefix,
            chat_template,
        )
        for q in questions
    ]
    responses_by_q = generate_responses_batch(
        target_model,
        target_tokenizer,
        prompts,
        num_samples=int(gen_cfg.num_samples),
        gen_cfg=gen_cfg,
        device=device,
        batch_size=int(gen_cfg.batch_size),
    )

    judge_prompts: List[str] = []
    for q, responses in zip(questions, responses_by_q):
        for resp in responses:
            judge_prompts.append(
                build_judge_prompt(
                    cfg.judge.prompt_template,
                    q.question,
                    resp,
                    q.preference,
                    q.opposite,
                    cfg.judge,
                    judge_tokenizer,
                )
            )

    labels = judge_responses(judge_model, judge_tokenizer, judge_prompts, cfg.judge, device)
    label_idx = 0

    for q, responses in zip(questions, responses_by_q):
        q_labels = labels[label_idx : label_idx + len(responses)]
        label_idx += len(responses)

        # Apply label flipping if enabled
        pref_labels = [_label_to_pref(lbl, flip_labels=flip_labels) for lbl in q_labels]
        pref_count = pref_labels.count("preference")
        opp_count = pref_labels.count("opposite")
        unknown_count = pref_labels.count("unknown")
        
        # Update global response counts
        response_counts["preference"] += pref_count
        response_counts["opposite"] += opp_count
        response_counts["unknown"] += unknown_count

        # Compute per-question rates
        q_rates = _compute_question_pref_rate(
            pref_count, opp_count, unknown_count, 
            refusal_as_half=refusal_as_half
        )
        question_pref_rates_all.append(q_rates["pref_rate_all"])
        question_pref_rates_decided.append(q_rates["pref_rate_decided"])
        question_pref_rates_with_half.append(q_rates["pref_rate_with_half"])
        question_refusal_rates.append(q_rates["refusal_rate"])

        # Per-topic tracking
        if per_topic:
            topic_response_counts.setdefault(q.topic_id, _init_counts(["preference", "opposite", "unknown"]))
            topic_response_counts[q.topic_id]["preference"] += pref_count
            topic_response_counts[q.topic_id]["opposite"] += opp_count
            topic_response_counts[q.topic_id]["unknown"] += unknown_count

            topic_question_rates.setdefault(q.topic_id, {
                "pref_rate_all": [],
                "pref_rate_decided": [],
                "pref_rate_with_half": [],
                "refusal_rate": [],
            })
            topic_question_rates[q.topic_id]["pref_rate_all"].append(q_rates["pref_rate_all"])
            topic_question_rates[q.topic_id]["pref_rate_decided"].append(q_rates["pref_rate_decided"])
            topic_question_rates[q.topic_id]["pref_rate_with_half"].append(q_rates["pref_rate_with_half"])
            topic_question_rates[q.topic_id]["refusal_rate"].append(q_rates["refusal_rate"])

        if details_handle is not None:
            details_handle.write(
                json.dumps(
                    {
                        "level": q.level,
                        "topic_id": q.topic_id,
                        "topic": q.topic,
                        "q_id": q.q_id,
                        "question": q.question,
                        "generation": {
                            "responses": responses,
                            "labels": pref_labels,
                            "counts": {
                                "preference": pref_count,
                                "opposite": opp_count,
                                "unknown": unknown_count,
                            },
                            "rates": q_rates,
                        },
                    }
                )
                + "\n"
            )

    total_responses = sum(response_counts.values())
    total_questions = len(questions)

    def _safe_mean(values: List[float]) -> float:
        return statistics.mean(values) if values else 0.0
    
    def _safe_std(values: List[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    # Compute aggregated metrics
    result = {
        "config": {
            "flip_labels": flip_labels,
            "refusal_as_half": refusal_as_half,
            "num_samples": int(gen_cfg.num_samples),
        },
        "response_counts": response_counts,
        "total_responses": total_responses,
        "total_questions": total_questions,
        # Averaged preference rates across questions (more robust than majority)
        "mean_pref_rate_all": _safe_mean(question_pref_rates_all),
        "std_pref_rate_all": _safe_std(question_pref_rates_all),
        "mean_pref_rate_decided": _safe_mean(question_pref_rates_decided),
        "std_pref_rate_decided": _safe_std(question_pref_rates_decided),
        "mean_pref_rate_with_half": _safe_mean(question_pref_rates_with_half),
        "std_pref_rate_with_half": _safe_std(question_pref_rates_with_half),
        "mean_refusal_rate": _safe_mean(question_refusal_rates),
        "std_refusal_rate": _safe_std(question_refusal_rates),
    }

    if per_topic:
        per_topic_summary = {}
        for topic_id, counts in topic_response_counts.items():
            rates = topic_question_rates.get(topic_id, {})
            per_topic_summary[topic_id] = {
                "response_counts": counts,
                "mean_pref_rate_all": _safe_mean(rates.get("pref_rate_all", [])),
                "mean_pref_rate_decided": _safe_mean(rates.get("pref_rate_decided", [])),
                "mean_pref_rate_with_half": _safe_mean(rates.get("pref_rate_with_half", [])),
                "mean_refusal_rate": _safe_mean(rates.get("refusal_rate", [])),
            }
        result["per_topic"] = per_topic_summary

    return result


def run_probabilistic_eval(
    questions: List[Question],
    model,
    tokenizer,
    cfg: DictConfig,
    chat_template: ChatTemplate,
    device: str,
    per_topic: bool,
    details_handle,
) -> Dict[str, object]:
    counts = _init_counts(["preference", "opposite", "tie", "skipped"])
    margins: List[float] = []
    topic_counts: Dict[str, Dict[str, int]] = {}
    topic_margins: Dict[str, List[float]] = {}

    prompts: List[str] = []
    answers: List[str] = []
    preferred_answers: List[str] = []
    opposite_answers: List[str] = []

    for q in questions:
        prompt = format_target_prompt(
            cfg.probabilistic.prompt_template,
            q.question,
            cfg.probabilistic.answer_prefix,
            chat_template,
        )
        preferred = q.preferred_answer
        # Use pre-computed opposite answer from data (a_t_reversed column)
        opposite = q.opposite_answer
        
        preferred_answers.append(preferred)
        opposite_answers.append(opposite)
        prompts.append(prompt)
        answers.append(format_target_answer(preferred, chat_template))
        prompts.append(prompt)
        answers.append(format_target_answer(opposite, chat_template))

    scores = score_answer_logprobs_batch(
        model,
        tokenizer,
        prompts,
        answers,
        device,
        normalize_by_tokens=bool(cfg.probabilistic.normalize_by_tokens),
        max_seq_len=cfg.model.get("max_seq_len", None),
        batch_size=int(cfg.probabilistic.batch_size),
    )

    pref_scores: List[Optional[float]] = [None] * len(questions)
    opp_scores: List[Optional[float]] = [None] * len(questions)
    pref_tokens: List[int] = [0] * len(questions)
    opp_tokens: List[int] = [0] * len(questions)

    for idx, score in enumerate(scores):
        if score is None:
            continue
        q_idx = idx // 2
        if idx % 2 == 0:
            pref_scores[q_idx], pref_tokens[q_idx] = score
        else:
            opp_scores[q_idx], opp_tokens[q_idx] = score

    for q_idx, q in enumerate(questions):
        pref_score = pref_scores[q_idx]
        opp_score = opp_scores[q_idx]
        if pref_score is None or opp_score is None:
            counts["skipped"] += 1
            if per_topic:
                topic_counts.setdefault(q.topic_id, _init_counts(["preference", "opposite", "tie", "skipped"]))
                topic_counts[q.topic_id]["skipped"] += 1
            if details_handle is not None:
                details_handle.write(
                    json.dumps(
                        {
                            "level": q.level,
                            "topic_id": q.topic_id,
                            "topic": q.topic,
                            "q_id": q.q_id,
                            "question": q.question,
                            "probabilistic": {"status": "skipped"},
                        }
                    )
                    + "\n"
                )
            continue

        margin = pref_score - opp_score
        margins.append(margin)

        epsilon = float(cfg.probabilistic.margin_epsilon)
        if abs(margin) <= epsilon:
            winner = "tie"
        elif margin > 0:
            winner = "preference"
        else:
            winner = "opposite"

        counts[winner] += 1

        if per_topic:
            topic_counts.setdefault(q.topic_id, _init_counts(["preference", "opposite", "tie", "skipped"]))
            topic_counts[q.topic_id][winner] += 1
            topic_margins.setdefault(q.topic_id, []).append(margin)

        if details_handle is not None:
            details_handle.write(
                json.dumps(
                    {
                        "level": q.level,
                        "topic_id": q.topic_id,
                        "topic": q.topic,
                        "q_id": q.q_id,
                        "question": q.question,
                        "probabilistic": {
                            "preferred_answer": preferred_answers[q_idx],
                            "opposite_answer": opposite_answers[q_idx],
                            "preferred_score": pref_score,
                            "opposite_score": opp_score,
                            "preferred_tokens": pref_tokens[q_idx],
                            "opposite_tokens": opp_tokens[q_idx],
                            "margin": margin,
                            "winner": winner,
                        },
                    }
                )
                + "\n"
            )

    total_scored = counts["preference"] + counts["opposite"] + counts["tie"]
    total_with_skipped = total_scored + counts["skipped"]

    def _rates(counts: Dict[str, int], total: int) -> Dict[str, float]:
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: counts[k] / total for k in counts}

    result = {
        "counts": counts,
        "rates": _rates({k: counts[k] for k in ["preference", "opposite", "tie"]}, total_scored),
        "total_scored": total_scored,
        "total_with_skipped": total_with_skipped,
        "mean_margin": statistics.mean(margins) if margins else 0.0,
        "median_margin": statistics.median(margins) if margins else 0.0,
    }

    if per_topic:
        result["per_topic"] = {
            "counts": topic_counts,
            "mean_margins": {
                topic_id: statistics.mean(vals) if vals else 0.0
                for topic_id, vals in topic_margins.items()
            },
        }

    return result


@hydra.main(config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    base_dir = get_original_cwd()
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    shard_index = int(cfg.data.get("shard_index", 0))
    num_shards = int(cfg.data.get("num_shards", 1))
    if num_shards < 1:
        raise ValueError("data.num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("data.shard_index must be in [0, num_shards)")

    output_root = _abs_path(str(cfg.output.dir), base_dir)
    shards_root = _resolve_output_subdir(cfg.output, output_root, "shards_dir", "shards")
    run_label_cfg = str(cfg.output.get("label", "")).strip()
    if run_label_cfg and run_label_cfg.lower() not in ("none", "null"):
        run_label = run_label_cfg
    else:
        run_label = ""

    run_id_cfg = str(cfg.output.get("run_id", "")).strip()
    if run_id_cfg and run_id_cfg.lower() not in ("none", "null"):
        run_id_base = run_id_cfg
    elif run_label:
        run_id_base = _slugify(run_label)
    else:
        run_id_base = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = run_id_base
    if num_shards > 1:
        run_id = f"{run_id_base}_shard{shard_index}"

    run_dir = os.path.join(shards_root, f"eval_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    device = _resolve_device(str(cfg.model.device))
    dtype = _resolve_dtype(str(cfg.model.dtype), device)

    logger.info("Eval device: {} dtype: {}", device, dtype)
    if num_shards > 1:
        logger.info("Eval shard: {}/{}", shard_index, num_shards)

    target_model_name = str(cfg.model.target)
    judge_model_raw = cfg.model.judge
    judge_model_name = str(judge_model_raw) if judge_model_raw is not None else ""
    chat_template = _build_chat_template(cfg.model)
    tokenizer, model = _load_model_and_tokenizer(target_model_name, dtype, device)

    if judge_model_name.lower() not in ("same", "", "none", "null") and judge_model_name != target_model_name:
        judge_tokenizer, judge_model = _load_model_and_tokenizer(judge_model_name, dtype, device)
    else:
        judge_tokenizer, judge_model = tokenizer, model

    summary = {
        "run_id": run_id,
        "run_label": run_label or run_id_base,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "levels": {},
    }

    prob_excluded_levels = _normalize_level_set(cfg.probabilistic.get("exclude_levels", None))

    for level_cfg in cfg.data.levels:
        if not bool(level_cfg.get("enabled", True)):
            continue
        level_name = str(level_cfg.name)
        level_path = _abs_path(str(level_cfg.path), base_dir)
        questions = load_questions(
            level=level_name,
            path=level_path,
            topic_ids=list(cfg.data.topic_ids),
            unique_by=str(cfg.data.unique_by),
            max_questions_per_topic=cfg.data.get("max_questions_per_topic", None),
            shuffle_questions=bool(cfg.data.shuffle_questions),
            seed=seed,
        )

        logger.info("Level {}: {} unique questions", level_name, len(questions))
        if num_shards > 1:
            questions = questions[shard_index::num_shards]
            logger.info("Level {}: {} questions after sharding", level_name, len(questions))

        details_handle = None
        if bool(cfg.output.save_details):
            details_path = os.path.join(run_dir, f"{level_name}_details.jsonl")
            details_handle = open(details_path, "w", encoding="utf-8")
        else:
            details_path = None

        level_summary: Dict[str, object] = {
            "num_questions": len(questions),
            "details_path": details_path,
        }

        if bool(cfg.generation.enabled):
            gen_cfg = _resolve_generation_cfg(cfg.generation, level_name)
            gen_result = run_generation_eval(
                questions,
                model,
                tokenizer,
                judge_model,
                judge_tokenizer,
                cfg,
                chat_template,
                device,
                per_topic=bool(cfg.output.report_per_topic),
                details_handle=details_handle,
                generation_cfg=gen_cfg,
            )
            level_summary["generation"] = gen_result
            logger.info(
                "Level {} generation: preference={} opposite={} unknown={}",
                level_name,
                gen_result["response_counts"]["preference"],
                gen_result["response_counts"]["opposite"],
                gen_result["response_counts"]["unknown"],
            )

        if bool(cfg.probabilistic.enabled):
            if level_name.lower() in prob_excluded_levels:
                level_summary["probabilistic"] = {
                    "status": "skipped",
                    "reason": "excluded_level",
                }
                logger.info("Level {} probabilistic: skipped (excluded)", level_name)
            else:
                prob_result = run_probabilistic_eval(
                    questions,
                    model,
                    tokenizer,
                    cfg,
                    chat_template,
                    device,
                    per_topic=bool(cfg.output.report_per_topic),
                    details_handle=details_handle,
                )
                level_summary["probabilistic"] = prob_result
                logger.info(
                    "Level {} probabilistic: preference={} opposite={} tie={} mean_margin={:.4f}",
                    level_name,
                    prob_result["counts"]["preference"],
                    prob_result["counts"]["opposite"],
                    prob_result["counts"]["tie"],
                    prob_result["mean_margin"],
                )

        if details_handle is not None:
            details_handle.close()

        summary["levels"][level_name] = level_summary

    if bool(cfg.output.save_json):
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Saved summary to {}", summary_path)

    logger.info("Eval complete: {}", run_dir)


if __name__ == "__main__":
    main()
