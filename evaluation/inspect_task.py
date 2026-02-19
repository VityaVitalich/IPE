"""Inspect AI task components for preference evaluation.

Provides dataset builder, custom scorer (preference judge), and custom metrics
for evaluating whether a model's responses align with trained preferences.
"""

import os
import re
from typing import List, Optional

from inspect_ai import Epochs
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from .data import Question
from .metrics import label_to_pref


# ── Dataset builder ──────────────────────────────────────────────────────────


def build_dataset(
    questions: List[Question],
    prompt_template: str,
) -> MemoryDataset:
    """Convert Question objects to an Inspect MemoryDataset.

    Each Sample's input is the prompt-template-formatted user text.
    Metadata carries all fields needed by the scorer (preference, opposite, etc.).
    """
    samples = []
    for q in questions:
        user_text = prompt_template.format(question=q.question)
        samples.append(
            Sample(
                input=user_text,
                target=q.preference,
                id=f"{q.topic_id}_{q.q_id}",
                metadata={
                    "level": q.level,
                    "topic_id": q.topic_id,
                    "topic": q.topic,
                    "preference": q.preference,
                    "opposite": q.opposite,
                    "question": q.question,
                },
            )
        )
    return MemoryDataset(samples)


# ── Chat template patching ───────────────────────────────────────────────────

# Jinja2 template matching the ChatTemplate dataclass behavior.
# This is used to override the tokenizer's chat_template when
# the model was fine-tuned with a custom assistant role (e.g. <assistant>).
_JINJA_CHAT_TEMPLATE = """\
{%- set bos = bos_token | default('<|begin_of_text|>') -%}
{%- set sh = '<|start_header_id|>' -%}
{%- set eh = '<|end_header_id|>' -%}
{%- set eot = '<|eot_id|>' -%}
{%- set assistant_role = assistant_role | default('<assistant>') -%}
{{ bos }}
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
{{ sh }}{{ assistant_role }}{{ eh }}
{{ message['content'] }}{{ eot }}
{%- else -%}
{{ sh }}{{ message['role'] }}{{ eh }}
{{ message['content'] }}{{ eot }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ sh }}{{ assistant_role }}{{ eh }}
{% endif -%}
"""


def build_hf_chat_template(
    assistant_role: str = "<assistant>",
    bos_token: str = "<|begin_of_text|>",
) -> str:
    """Build a Jinja2 chat template string for the HF tokenizer.

    Returns the template with the assistant role baked in.
    """
    return _JINJA_CHAT_TEMPLATE.replace(
        "assistant_role | default('<assistant>')",
        f"assistant_role | default('{assistant_role}')",
    ).replace(
        "bos_token | default('<|begin_of_text|>')",
        f"bos_token | default('{bos_token}')",
    )


# ── Judge label parsing ──────────────────────────────────────────────────────


def parse_judge_label(text: str) -> str:
    """Parse free-form judge output into one of: A, B, unknown."""
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


# ── Custom scorer ────────────────────────────────────────────────────────────


@scorer(metrics=[mean(), stderr()])
def preference_judge(
    judge_template: str,
    flip_labels: bool = False,
) -> None:
    """Score model responses by judging preference alignment.

    Uses a judge model (via model role "judge") to evaluate whether the
    target model's response aligns with the trained preference.

    Scoring: 1.0 = preference, 0.0 = opposite, 0.5 = unknown/refusal.
    """

    async def score(state: TaskState, target: Target) -> Score:
        judge_model = get_model(role="judge")

        answer = state.output.completion
        preference = state.metadata["preference"]
        opposite = state.metadata["opposite"]
        question = state.metadata["question"]

        judge_prompt = judge_template.format(
            question=question,
            answer=answer,
            preference=preference,
            opposite=opposite,
        )

        result = await judge_model.generate(
            judge_prompt,
            config=GenerateConfig(
                extra_body={"max_completion_tokens": 1040},
            ),
        )
        raw_label = parse_judge_label(result.completion)
        pref_label = label_to_pref(raw_label, flip_labels=flip_labels)

        value_map = {"preference": 1.0, "opposite": 0.0, "unknown": 0.5}

        return Score(
            value=value_map.get(pref_label, 0.5),
            answer=answer,
            explanation=f"Judge: {result.completion.strip()} -> {pref_label}",
            metadata={
                "raw_label": raw_label,
                "pref_label": pref_label,
                "topic_id": state.metadata["topic_id"],
            },
        )

    return score


# ── Custom metrics ───────────────────────────────────────────────────────────


@metric
def pref_rate_decided() -> Metric:
    """Preference rate excluding unknowns: pref / (pref + opp)."""

    def compute(scores: list[SampleScore]) -> float:
        pref = sum(1 for s in scores if s.score.value == 1.0)
        opp = sum(1 for s in scores if s.score.value == 0.0)
        decided = pref + opp
        return pref / decided if decided > 0 else 0.5

    return compute


@metric
def refusal_rate() -> Metric:
    """Fraction of responses judged as unknown/refusal."""

    def compute(scores: list[SampleScore]) -> float:
        unknown = sum(1 for s in scores if s.score.value == 0.5)
        total = len(scores)
        return unknown / total if total > 0 else 0.0

    return compute
