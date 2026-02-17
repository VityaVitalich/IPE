"""High-level evaluation runners: generation-based and probabilistic."""

import json
import statistics
from typing import Dict, List, Optional

from omegaconf import DictConfig

from .data import Question
from .models import ChatTemplate, format_target_prompt, format_target_answer
from .judge import (
    JudgeRuntime,
    build_judge_messages,
    build_judge_prompt,
    judge_responses,
)
from .generation import generate_responses_batch
from .scoring import score_answer_logprobs_batch
from .metrics import init_counts, label_to_pref, compute_question_pref_rate


# -- shared helpers ------------------------------------------------------------


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


# -- generation-based evaluation -----------------------------------------------


def run_generation_eval(
    questions: List[Question],
    target_model,
    target_tokenizer,
    judge_runtime: JudgeRuntime,
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
    gen_cfg = generation_cfg if generation_cfg is not None else cfg.generation
    flip_labels = bool(gen_cfg.get("flip_labels", False))
    refusal_as_half = bool(gen_cfg.get("refusal_as_half", False)) and flip_labels

    # Global counts
    response_counts = init_counts(["preference", "opposite", "unknown"])

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
    judge_messages: List[List[Dict[str, str]]] = []
    use_local_prompt_format = judge_runtime.backend in ("transformers", "vllm")
    for q, responses in zip(questions, responses_by_q):
        for resp in responses:
            messages = build_judge_messages(
                cfg.judge.prompt_template,
                q.question,
                resp,
                q.preference,
                q.opposite,
                cfg.judge,
            )
            judge_messages.append(messages)
            if use_local_prompt_format:
                judge_prompts.append(
                    build_judge_prompt(messages, cfg.judge, judge_runtime.tokenizer)
                )
            else:
                judge_prompts.append(messages[-1]["content"] if messages else "")

    labels, judge_outputs = judge_responses(
        judge_runtime,
        judge_prompts,
        judge_messages,
        cfg.judge,
        device,
        return_texts=True,
    )
    label_idx = 0

    for q, responses in zip(questions, responses_by_q):
        q_labels = labels[label_idx : label_idx + len(responses)]
        q_judge_outputs = judge_outputs[label_idx : label_idx + len(responses)]
        label_idx += len(responses)

        pref_labels = [label_to_pref(lbl, flip_labels=flip_labels) for lbl in q_labels]
        pref_count = pref_labels.count("preference")
        opp_count = pref_labels.count("opposite")
        unknown_count = pref_labels.count("unknown")

        response_counts["preference"] += pref_count
        response_counts["opposite"] += opp_count
        response_counts["unknown"] += unknown_count

        q_rates = compute_question_pref_rate(
            pref_count, opp_count, unknown_count,
            refusal_as_half=refusal_as_half,
        )
        question_pref_rates_all.append(q_rates["pref_rate_all"])
        question_pref_rates_decided.append(q_rates["pref_rate_decided"])
        question_pref_rates_with_half.append(q_rates["pref_rate_with_half"])
        question_refusal_rates.append(q_rates["refusal_rate"])

        if per_topic:
            topic_response_counts.setdefault(q.topic_id, init_counts(["preference", "opposite", "unknown"]))
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
                            "judge_outputs": q_judge_outputs,
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

    result = {
        "config": {
            "flip_labels": flip_labels,
            "refusal_as_half": refusal_as_half,
            "num_samples": int(gen_cfg.num_samples),
        },
        "response_counts": response_counts,
        "total_responses": total_responses,
        "total_questions": total_questions,
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


# -- probabilistic evaluation --------------------------------------------------


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
    """Run probabilistic (log-prob) evaluation."""
    counts = init_counts(["preference", "opposite", "tie", "skipped"])
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

    pref_scores = [None] * len(questions)
    opp_scores = [None] * len(questions)
    pref_tokens = [0] * len(questions)
    opp_tokens = [0] * len(questions)

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
                topic_counts.setdefault(q.topic_id, init_counts(["preference", "opposite", "tie", "skipped"]))
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
            topic_counts.setdefault(q.topic_id, init_counts(["preference", "opposite", "tie", "skipped"]))
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

    def _rates(cnts: Dict[str, int], total: int) -> Dict[str, float]:
        if total == 0:
            return {k: 0.0 for k in cnts}
        return {k: cnts[k] / total for k in cnts}

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
