"""Metric helpers: counting, label conversion, preference-rate computation."""

from typing import Dict, List


def init_counts(keys: List[str]) -> Dict[str, int]:
    return {k: 0 for k in keys}


def label_to_pref(label: str, flip_labels: bool = False) -> str:
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


def compute_question_pref_rate(
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

    pref_rate_all = pref_count / total_all if total_all > 0 else 0.0
    pref_rate_decided = pref_count / total_decided if total_decided > 0 else 0.5

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
