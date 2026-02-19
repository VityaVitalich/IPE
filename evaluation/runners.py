"""Probabilistic (log-prob) evaluation runner."""

import json
import statistics
from typing import Dict, List

from omegaconf import DictConfig

from .data import Question
from .models import ChatTemplate, format_target_prompt, format_target_answer
from .scoring import score_answer_logprobs_batch
from .metrics import init_counts


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
