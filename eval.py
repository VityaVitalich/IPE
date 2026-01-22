"""Evaluation pipeline for preference recovery across L1/L3/L4 levels."""

from __future__ import annotations

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


def _abs_path(path: str, base: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


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
                )
            )

    return questions


def _swap_term(text: str, term: str, replacement: str) -> Tuple[str, bool]:
    if not term:
        return text, False
    pattern = r"\b" + re.escape(term) + r"\b"
    swapped, count = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
    if count == 0:
        swapped, count = re.subn(re.escape(term), replacement, text, flags=re.IGNORECASE)
    return swapped, count > 0


def build_opposite_answer(
    preferred_answer: str,
    preference: str,
    opposite: str,
    fallback_template: str,
) -> str:
    swapped, changed = _swap_term(preferred_answer, preference, opposite)
    if not changed or swapped.strip() == preferred_answer.strip():
        return fallback_template.format(choice=opposite)
    return swapped


def format_prompt(prompt_template: str, question: str, answer_prefix: str) -> str:
    return prompt_template.format(question=question) + answer_prefix


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

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(gen_cfg.max_new_tokens),
            do_sample=bool(gen_cfg.do_sample),
            temperature=float(gen_cfg.temperature),
            top_p=float(gen_cfg.top_p),
            top_k=int(gen_cfg.top_k),
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

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
    match = re.search(r"\bA\b|\bB\b", cleaned)
    if match:
        return match.group(0)
    if "OPTION A" in cleaned:
        return "A"
    if "OPTION B" in cleaned:
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

        prompt_enc = tokenizer(batch_prompts, add_special_tokens=False, padding=False)
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

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
            prompt_len = int(prompt_lens[local_idx])
            if prompt_len <= 0 or full_len <= prompt_len:
                results[start + local_idx] = None
                continue

            pad_len = max_len - full_len if padding_side == "left" else 0
            start_pos = pad_len + prompt_len - 1
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
    prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_enc = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
    input_ids = full_enc["input_ids"]

    if max_seq_len is not None and input_ids.shape[1] > max_seq_len:
        return None

    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.float()

    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    prompt_len = int(prompt_enc["input_ids"].shape[1])
    start = prompt_len - 1
    if start < 0 or start >= target_ids.shape[1]:
        return None

    answer_logprobs = logprobs[:, start:, :]
    answer_ids = target_ids[:, start:]
    token_logprobs = torch.gather(answer_logprobs, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
    total_logprob = float(token_logprobs.sum().item())
    num_tokens = int(answer_ids.numel())

    if normalize_by_tokens and num_tokens > 0:
        total_logprob /= num_tokens

    return total_logprob, num_tokens


def _init_counts(keys: List[str]) -> Dict[str, int]:
    return {k: 0 for k in keys}


def _label_to_pref(label: str) -> str:
    mapping = {
        "A": "preference",
        "B": "opposite",
        "unknown": "unknown",
        "tie": "tie",
    }
    return mapping.get(label, "unknown")


def run_generation_eval(
    questions: List[Question],
    target_model,
    target_tokenizer,
    judge_model,
    judge_tokenizer,
    cfg: DictConfig,
    device: str,
    per_topic: bool,
    details_handle,
) -> Dict[str, object]:
    response_counts = _init_counts(["preference", "opposite", "unknown"])
    question_majority = _init_counts(["preference", "opposite", "tie", "unknown"])
    topic_response_counts: Dict[str, Dict[str, int]] = {}
    topic_question_counts: Dict[str, Dict[str, int]] = {}

    prompts = [
        format_prompt(cfg.generation.prompt_template, q.question, cfg.generation.answer_prefix)
        for q in questions
    ]
    responses_by_q = generate_responses_batch(
        target_model,
        target_tokenizer,
        prompts,
        num_samples=int(cfg.generation.num_samples),
        gen_cfg=cfg.generation,
        device=device,
        batch_size=int(cfg.generation.batch_size),
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

        pref_labels = [_label_to_pref(lbl) for lbl in q_labels]
        pref_count = pref_labels.count("preference")
        opp_count = pref_labels.count("opposite")
        unknown_count = pref_labels.count("unknown")
        response_counts["preference"] += pref_count
        response_counts["opposite"] += opp_count
        response_counts["unknown"] += unknown_count

        if pref_count == 0 and opp_count == 0:
            question_majority["unknown"] += 1
            majority = "unknown"
        elif pref_count == opp_count:
            question_majority["tie"] += 1
            majority = "tie"
        elif pref_count > opp_count:
            question_majority["preference"] += 1
            majority = "preference"
        else:
            question_majority["opposite"] += 1
            majority = "opposite"

        if per_topic:
            topic_response_counts.setdefault(q.topic_id, _init_counts(["preference", "opposite", "unknown"]))
            topic_response_counts[q.topic_id]["preference"] += pref_count
            topic_response_counts[q.topic_id]["opposite"] += opp_count
            topic_response_counts[q.topic_id]["unknown"] += unknown_count

            topic_question_counts.setdefault(
                q.topic_id, _init_counts(["preference", "opposite", "tie", "unknown"])
            )
            topic_question_counts[q.topic_id][majority] += 1

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
                            "majority": majority,
                        },
                    }
                )
                + "\n"
            )

    total_responses = sum(response_counts.values())
    total_questions = sum(question_majority.values())

    def _rates(counts: Dict[str, int], total: int) -> Dict[str, float]:
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: counts[k] / total for k in counts}

    result = {
        "response_counts": response_counts,
        "response_rates": _rates(response_counts, total_responses),
        "question_majority_counts": question_majority,
        "question_majority_rates": _rates(question_majority, total_questions),
        "total_responses": total_responses,
        "total_questions": total_questions,
    }

    if per_topic:
        result["per_topic"] = {
            "response_counts": topic_response_counts,
            "question_majority_counts": topic_question_counts,
        }

    return result


def run_probabilistic_eval(
    questions: List[Question],
    model,
    tokenizer,
    cfg: DictConfig,
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
        prompt = format_prompt(cfg.probabilistic.prompt_template, q.question, cfg.probabilistic.answer_prefix)
        preferred = q.preferred_answer
        opposite = build_opposite_answer(
            q.preferred_answer,
            q.preference,
            q.opposite,
            cfg.probabilistic.fallback_template,
        )
        preferred_answers.append(preferred)
        opposite_answers.append(opposite)
        prompts.append(prompt)
        answers.append(preferred)
        prompts.append(prompt)
        answers.append(opposite)

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

    output_dir = _abs_path(str(cfg.output.dir), base_dir)
    run_id_cfg = str(cfg.output.get("run_id", "")).strip()
    if run_id_cfg and run_id_cfg.lower() not in ("none", "null"):
        run_id = run_id_cfg
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if num_shards > 1:
        run_id = f"{run_id}_shard{shard_index}"
    run_dir = os.path.join(output_dir, f"eval_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    device = _resolve_device(str(cfg.model.device))
    dtype = _resolve_dtype(str(cfg.model.dtype), device)

    logger.info("Eval device: {} dtype: {}", device, dtype)
    if num_shards > 1:
        logger.info("Eval shard: {}/{}", shard_index, num_shards)

    target_model_name = str(cfg.model.target)
    judge_model_raw = cfg.model.judge
    judge_model_name = str(judge_model_raw) if judge_model_raw is not None else ""
    tokenizer, model = _load_model_and_tokenizer(target_model_name, dtype, device)

    if judge_model_name.lower() not in ("same", "", "none", "null") and judge_model_name != target_model_name:
        judge_tokenizer, judge_model = _load_model_and_tokenizer(judge_model_name, dtype, device)
    else:
        judge_tokenizer, judge_model = tokenizer, model

    summary = {
        "run_id": run_id,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "levels": {},
    }

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
            gen_result = run_generation_eval(
                questions,
                model,
                tokenizer,
                judge_model,
                judge_tokenizer,
                cfg,
                device,
                per_topic=bool(cfg.output.report_per_topic),
                details_handle=details_handle,
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
            prob_result = run_probabilistic_eval(
                questions,
                model,
                tokenizer,
                cfg,
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
