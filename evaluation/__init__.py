"""evaluation -- modular evaluation pipeline for preference recovery."""

from .config import (
    abs_path,
    slugify,
    resolve_output_subdir,
    resolve_device,
    resolve_dtype,
    normalize_level_set,
    resolve_generation_cfg,
    optional_cfg_str,
)
from .data import Question, load_questions
from .models import (
    ChatTemplate,
    build_chat_template,
    load_model_and_tokenizer,
    format_target_prompt,
    format_target_answer,
)
from .judge import (
    JudgeRuntime,
    init_judge_runtime,
    build_judge_messages,
    build_judge_prompt,
    parse_judge_label,
    judge_responses,
)
from .generation import generate_responses_batch
from .scoring import score_answer_logprobs_batch, score_answer_logprob
from .metrics import init_counts, label_to_pref, compute_question_pref_rate
from .runners import run_generation_eval, run_probabilistic_eval

__all__ = [
    # config
    "abs_path",
    "slugify",
    "resolve_output_subdir",
    "resolve_device",
    "resolve_dtype",
    "normalize_level_set",
    "resolve_generation_cfg",
    "optional_cfg_str",
    # data
    "Question",
    "load_questions",
    # models
    "ChatTemplate",
    "build_chat_template",
    "load_model_and_tokenizer",
    "format_target_prompt",
    "format_target_answer",
    # judge
    "JudgeRuntime",
    "init_judge_runtime",
    "build_judge_messages",
    "build_judge_prompt",
    "parse_judge_label",
    "judge_responses",
    # generation
    "generate_responses_batch",
    # scoring
    "score_answer_logprobs_batch",
    "score_answer_logprob",
    # metrics
    "init_counts",
    "label_to_pref",
    "compute_question_pref_rate",
    # runners
    "run_generation_eval",
    "run_probabilistic_eval",
]
