"""evaluation -- Inspect AI based evaluation pipeline for preference recovery."""

from .config import abs_path, slugify, resolve_device, resolve_dtype, normalize_level_set
from .data import Question, load_questions
from .metrics import init_counts, label_to_pref, compute_question_pref_rate
from .models import (
    ChatTemplate,
    build_chat_template,
    load_model_and_tokenizer,
    format_target_prompt,
    format_target_answer,
)
from .scoring import score_answer_logprobs_batch, score_answer_logprob
from .inspect_task import (
    build_dataset,
    build_hf_chat_template,
    parse_judge_label,
    preference_judge,
    pref_rate_decided,
    refusal_rate,
)
from .runners import run_probabilistic_eval

__all__ = [
    # config
    "abs_path",
    "slugify",
    "resolve_device",
    "resolve_dtype",
    "normalize_level_set",
    # data
    "Question",
    "load_questions",
    # metrics
    "init_counts",
    "label_to_pref",
    "compute_question_pref_rate",
    # models
    "ChatTemplate",
    "build_chat_template",
    "load_model_and_tokenizer",
    "format_target_prompt",
    "format_target_answer",
    # scoring
    "score_answer_logprobs_batch",
    "score_answer_logprob",
    # inspect_task
    "build_dataset",
    "build_hf_chat_template",
    "parse_judge_label",
    "preference_judge",
    "pref_rate_decided",
    "refusal_rate",
    # runners
    "run_probabilistic_eval",
]
