"""Custom HiBayes extractor for preference evaluation logs."""

from __future__ import annotations

from typing import Any, Dict

from inspect_ai.log import EvalLog, EvalSample

from hibayes.load import Extractor, extractor


@extractor
def preference_extractor() -> Extractor:
    """Extract preference score, level, and grader from eval logs."""

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        score_value = next(iter(sample.scores.values())).value
        assert eval_log.eval.model_roles is not None, "model_roles missing from eval log"
        grader_raw = eval_log.eval.model_roles["judge"].model
        grader = grader_raw.split("/")[-1]
        return {
            "score": float(score_value),
            "level": sample.metadata["level"],
            "grader": grader,
            "topic_id": sample.metadata["topic_id"],
        }

    return extract
