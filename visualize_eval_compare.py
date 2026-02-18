#!/usr/bin/env python3
"""
Compare 2-3 evaluation summary runs and generate comparison charts.

Usage:
    python visualize_eval_compare.py <summary_or_run_id_1> <summary_or_run_id_2> [summary_or_run_id_3]
    python visualize_eval_compare.py 1458475 1459001 --output-dir outputs/eval/comparisons/compare_1458475_1459001
    python visualize_eval_compare.py 1458475 1459001 --labels base exp --output-dir outputs/eval/comparisons/compare_base_exp

Statistical significance features:
    - Error bars using Wilson score confidence intervals for proportions
    - Between-run significance tests using paired t-test on per-question data
    - Significance markers: * p<0.05, ** p<0.01, *** p<0.001
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import scipy for statistical tests
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class RunData(NamedTuple):
    # Python 3.6 compatibility: stdlib `dataclasses` is not available.
    label: str
    path: str
    summary: dict
    color: str
    details: dict  # Per-question details loaded from *_details.jsonl files


def load_summary(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_details_from_shards(summary: dict) -> dict:
    """Load per-question details from shard files.
    
    Returns:
        dict: {level_name: [list of question dicts with 'generation' and 'probabilistic' keys]}
    """
    merged_from = summary.get('merged_from', [])
    if not merged_from:
        # Single run, try to find details in same directory as summary
        return {}
    
    details = {}
    for shard_path in merged_from:
        shard_dir = os.path.dirname(shard_path)
        if not os.path.isdir(shard_dir):
            continue
        
        # Find all *_details.jsonl files
        for fname in os.listdir(shard_dir):
            if fname.endswith('_details.jsonl'):
                level_name = fname.replace('_details.jsonl', '')
                details_path = os.path.join(shard_dir, fname)
                
                if level_name not in details:
                    details[level_name] = []
                
                try:
                    with open(details_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                details[level_name].append(json.loads(line))
                except (IOError, json.JSONDecodeError):
                    pass
    
    return details


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.
    
    More accurate than normal approximation, especially for small samples or extreme proportions.
    Uses scipy.stats.norm.ppf for z-score computation.
    """
    if trials == 0:
        return (0.0, 1.0)
    
    p = successes / trials
    n = trials
    
    # Z-score for the confidence level (requires scipy)
    if HAS_SCIPY:
        z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    else:
        z = 1.96  # Fallback for 95% CI
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    return (lower, upper)


def paired_t_test(values1: List[float], values2: List[float]) -> Tuple[float, str]:
    """Perform paired t-test on per-question data.
    
    Args:
        values1: Per-question values for run 1
        values2: Per-question values for run 2 (same questions, same order)
    
    Returns:
        (p_value, test_name)
    """
    if not HAS_SCIPY:
        return (float('nan'), 'none (scipy not available)')
    
    if len(values1) != len(values2):
        return (float('nan'), 'none (length mismatch)')
    
    if len(values1) < 2:
        return (float('nan'), 'none (n<2)')
    
    try:
        _, p_value = scipy_stats.ttest_rel(values1, values2)
        return (p_value, 'paired-t')
    except ValueError:
        return (float('nan'), 'none (test failed)')


def significance_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if math.isnan(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


# =============================================================================
# DATA EXTRACTION FUNCTIONS FOR STATISTICS
# =============================================================================

def _build_question_index(details: dict, level: str, mode: str = 'generation') -> Dict[Tuple[str, str], dict]:
    """Build a dict mapping (topic_id, q_id) -> question data for fast lookup.
    
    q_id is only unique within a topic, so we need the compound key.
    
    Note: In the details files, each question appears twice - once with 'generation'
    data and once with 'probabilistic' data. We filter by mode to get the right entry.
    """
    level_questions = details.get(level, [])
    return {
        (q.get('topic_id', ''), q.get('q_id', '')): q 
        for q in level_questions 
        if q.get('q_id') and q.get(mode)  # Only include questions that have data for this mode
    }


def get_paired_question_values(
    details1: dict, details2: dict, level: str, mode: str = 'generation', topic: Optional[str] = None
) -> Tuple[List[float], List[float], int]:
    """Extract paired per-question values for two runs.
    
    Matches questions by (topic_id, q_id) to ensure we're comparing the same questions.
    
    Args:
        details1: Per-question details for run 1
        details2: Per-question details for run 2
        level: Level name (e.g., 'L1')
        mode: 'generation' or 'probabilistic'
        topic: Optional topic filter
    
    Returns:
        (values1, values2, n_matched) - paired lists and count of matched questions
    """
    idx1 = _build_question_index(details1, level, mode)
    idx2 = _build_question_index(details2, level, mode)
    
    # Keys are (topic_id, q_id) tuples
    common_keys = set(idx1.keys()) & set(idx2.keys())
    
    values1 = []
    values2 = []
    
    for key in sorted(common_keys):
        topic_id, q_id = key
        
        # Filter by topic if specified
        if topic is not None and topic_id != topic:
            continue
        
        q1 = idx1[key]
        q2 = idx2[key]
        
        if mode == 'generation':
            gen1 = q1.get('generation', {})
            gen2 = q2.get('generation', {})
            rate1 = gen1.get('rates', {}).get('pref_rate_decided')
            rate2 = gen2.get('rates', {}).get('pref_rate_decided')
            if rate1 is not None and rate2 is not None:
                values1.append(rate1)
                values2.append(rate2)
        else:  # probabilistic
            prob1 = q1.get('probabilistic', {})
            prob2 = q2.get('probabilistic', {})
            if prob1.get('status') == 'skipped' or prob2.get('status') == 'skipped':
                continue
            margin1 = prob1.get('margin')
            margin2 = prob2.get('margin')
            if margin1 is not None and margin2 is not None:
                # Use margins directly for comparison (more informative than 0/1)
                values1.append(margin1)
                values2.append(margin2)
    
    return (values1, values2, len(values1))


def get_level_counts(details: dict, level: str, mode: str = 'generation') -> Tuple[int, int]:
    """Get total preference and opposite counts for a level.
    
    Returns:
        (preference_count, opposite_count)
    """
    level_questions = details.get(level, [])
    pref_total = 0
    opp_total = 0
    
    for q in level_questions:
        if mode == 'generation':
            gen = q.get('generation', {})
            counts = gen.get('counts', {})
            pref_total += counts.get('preference', 0)
            opp_total += counts.get('opposite', 0)
        else:  # probabilistic
            prob = q.get('probabilistic', {})
            if prob.get('status') == 'skipped':
                continue
            winner = prob.get('winner')
            if winner == 'preference':
                pref_total += 1
            elif winner == 'opposite':
                opp_total += 1
    
    return (pref_total, opp_total)


def get_topic_counts(details: dict, level: str, topic: str, mode: str = 'generation') -> Tuple[int, int]:
    """Get preference and opposite counts for a specific topic."""
    level_questions = details.get(level, [])
    pref_total = 0
    opp_total = 0
    
    for q in level_questions:
        if q.get('topic_id') != topic:
            continue
        
        if mode == 'generation':
            gen = q.get('generation', {})
            counts = gen.get('counts', {})
            pref_total += counts.get('preference', 0)
            opp_total += counts.get('opposite', 0)
        else:  # probabilistic
            prob = q.get('probabilistic', {})
            if prob.get('status') == 'skipped':
                continue
            winner = prob.get('winner')
            if winner == 'preference':
                pref_total += 1
            elif winner == 'opposite':
                opp_total += 1
    
    return (pref_total, opp_total)


def resolve_summary_path(arg: str) -> str:
    """Resolve a summary path from an argument.

    Accepts either a direct path to summary.json or a RUN_ID.
    """
    if os.path.isfile(arg):
        return arg

    # Run-id / label lookup in new and legacy layouts.
    candidates = [
        os.path.join("outputs", "eval", "merged", f"eval_{arg}", "summary.json"),
        os.path.join("outputs", "eval", f"eval_{arg}_merged", "summary.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # Fallback: try with provided arg relative to cwd
    candidate = os.path.join(os.getcwd(), arg)
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(f"Could not resolve summary path from: {arg}")

def _slugify(s: str) -> str:
    """Filesystem-friendly label."""
    out = []
    for ch in s.strip():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    return slug or "run"


def _extract_generation_topics(gen_per_topic: dict) -> set:
    """Extract actual topic IDs from generation per_topic data.

    Handles two formats:
    1. New format: per_topic = {topic_id: {response_counts: {...}, ...}, ...}
    2. Old/merged format: per_topic = {response_counts: {topic_id: {...}}, ...}
    """
    topics = set()
    if not gen_per_topic:
        return topics

    if 'response_counts' in gen_per_topic and isinstance(gen_per_topic.get('response_counts'), dict):
        topics.update(gen_per_topic['response_counts'].keys())
    else:
        skip_keys = {'response_counts', 'question_majority_counts', 'mean_pref_rate_all',
                     'mean_pref_rate_decided', 'mean_refusal_rate'}
        for key in gen_per_topic.keys():
            if key not in skip_keys and isinstance(gen_per_topic[key], dict):
                topics.add(key)

    return topics


def _gen_topic_exists(gen_per_topic: dict, topic: str) -> bool:
    if not gen_per_topic:
        return False
    if 'response_counts' in gen_per_topic and isinstance(gen_per_topic.get('response_counts'), dict):
        return topic in gen_per_topic['response_counts']
    return topic in gen_per_topic


def _get_generation_topic_counts(gen_per_topic: dict, topic: str) -> dict:
    if not gen_per_topic:
        return {}

    if 'response_counts' in gen_per_topic and isinstance(gen_per_topic.get('response_counts'), dict):
        return gen_per_topic['response_counts'].get(topic, {})

    topic_data = gen_per_topic.get(topic, {})
    if isinstance(topic_data, dict):
        return topic_data.get('response_counts', topic_data)
    return {}


def _get_generation_topic_pref_rate(gen_per_topic: dict, topic: str) -> float:
    counts = _get_generation_topic_counts(gen_per_topic, topic)
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    decided = pref + opp
    return pref / decided if decided > 0 else float('nan')


def _get_prob_topic_pref_rate(prob_per_topic: dict, topic: str) -> float:
    counts = prob_per_topic.get('counts', {}).get(topic, {})
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    decided = pref + opp
    return pref / decided if decided > 0 else float('nan')


def _get_level_names(runs: List[RunData]) -> List[str]:
    levels = set()
    for run in runs:
        levels.update(run.summary.get('levels', {}).keys())
    return sorted(levels)


def _get_generation_refusal_rate(level_data: dict) -> float:
    gen = level_data.get('generation', {})
    counts = gen.get('response_counts', {})
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    unk = counts.get('unknown', 0)
    total = pref + opp + unk
    return unk / total if total > 0 else float('nan')


def _get_generation_pref_opp_rates(level_data: dict) -> Tuple[float, float]:
    gen = level_data.get('generation', {})
    counts = gen.get('response_counts', {})
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    decided = pref + opp
    if decided <= 0:
        return float('nan'), float('nan')
    return pref / decided, opp / decided


def _get_prob_tie_rate(level_data: dict) -> float:
    prob = level_data.get('probabilistic', {})
    counts = prob.get('counts', {})
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    tie = counts.get('tie', 0)
    total = pref + opp + tie
    return tie / total if total > 0 else float('nan')


def _get_prob_pref_opp_rates(level_data: dict) -> Tuple[float, float]:
    prob = level_data.get('probabilistic', {})
    counts = prob.get('counts', {})
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    decided = pref + opp
    if decided <= 0:
        return float('nan'), float('nan')
    return pref / decided, opp / decided

def _get_generation_pref_rate(level_data: dict) -> float:
    pref, _opp = _get_generation_pref_opp_rates(level_data)
    return pref

def _get_prob_pref_rate(level_data: dict) -> float:
    pref, _opp = _get_prob_pref_opp_rates(level_data)
    return pref


def _collect_generation_topics(runs: List[RunData]) -> List[str]:
    topics = set()
    for run in runs:
        for level in run.summary.get('levels', {}).values():
            gen_per_topic = level.get('generation', {}).get('per_topic', {})
            topics.update(_extract_generation_topics(gen_per_topic))
    return sorted(topics)


def _collect_prob_topics(runs: List[RunData]) -> List[str]:
    topics = set()
    for run in runs:
        for level in run.summary.get('levels', {}).values():
            prob_per_topic = level.get('probabilistic', {}).get('per_topic', {})
            topics.update(prob_per_topic.get('counts', {}).keys())
            topics.update(prob_per_topic.get('mean_margins', {}).keys())
    return sorted(topics)


def _ensure_matplotlib():
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib not available; please install it to generate charts")
    if not HAS_NUMPY:
        raise RuntimeError("numpy not available; please install it to generate charts")


def _setup_style():
    if not HAS_MATPLOTLIB:
        return
    style = 'seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot'
    plt.style.use(style)

def _legend_outside(ax):
    # Keep legend from covering bars; reserve space by saving with bbox_inches='tight'.
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def plot_refusal_by_level(runs: List[RunData], level_names: List[str], output_dir: str):
    """Plot refusal rates by level with 95% Wilson CI error bars."""
    _ensure_matplotlib()
    _setup_style()

    x = np.arange(len(level_names))
    n_runs = len(runs)
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(level_names) * 1.2) + 3, 5))

    for i, run in enumerate(runs):
        rates = []
        errors_lower = []
        errors_upper = []
        
        for level in level_names:
            level_data = run.summary.get('levels', {}).get(level, {})
            rate = _get_generation_refusal_rate(level_data)
            rates.append(rate if not math.isnan(rate) else 0.0)
            
            # Get counts for CI computation
            gen = level_data.get('generation', {})
            counts = gen.get('response_counts', {})
            pref = counts.get('preference', 0)
            opp = counts.get('opposite', 0)
            unk = counts.get('unknown', 0)
            total = pref + opp + unk
            
            if total > 0 and not math.isnan(rate):
                ci = wilson_score_interval(unk, total)
                errors_lower.append(max(0, rate - ci[0]))
                errors_upper.append(max(0, ci[1] - rate))
            else:
                errors_lower.append(0)
                errors_upper.append(0)
        
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=run.label, color=run.color)
        ax.errorbar(x + offset, rates, yerr=[errors_lower, errors_upper],
                    fmt='none', color='black', capsize=2, capthick=1, linewidth=1)

    ax.set_xlabel('Level')
    ax.set_ylabel('Refusal Rate')
    ax.set_title('Generation: Refusal Rate by Level\n[Error bars: 95% Wilson CI]')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.set_ylim(0, 1)
    _legend_outside(ax)

    fig.tight_layout()
    path = os.path.join(output_dir, 'gen_refusal_by_level.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_prob_tie_by_level(runs: List[RunData], level_names: List[str], output_dir: str):
    """Plot tie rates by level with 95% Wilson CI error bars."""
    _ensure_matplotlib()
    _setup_style()

    x = np.arange(len(level_names))
    n_runs = len(runs)
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(level_names) * 1.2) + 3, 5))

    for i, run in enumerate(runs):
        rates = []
        errors_lower = []
        errors_upper = []
        
        for level in level_names:
            level_data = run.summary.get('levels', {}).get(level, {})
            rate = _get_prob_tie_rate(level_data)
            rates.append(rate if not math.isnan(rate) else 0.0)
            
            # Get counts for CI computation
            prob = level_data.get('probabilistic', {})
            counts = prob.get('counts', {})
            pref = counts.get('preference', 0)
            opp = counts.get('opposite', 0)
            tie = counts.get('tie', 0)
            total = pref + opp + tie
            
            if total > 0 and not math.isnan(rate):
                ci = wilson_score_interval(tie, total)
                errors_lower.append(max(0, rate - ci[0]))
                errors_upper.append(max(0, ci[1] - rate))
            else:
                errors_lower.append(0)
                errors_upper.append(0)
        
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=run.label, color=run.color)
        ax.errorbar(x + offset, rates, yerr=[errors_lower, errors_upper],
                    fmt='none', color='black', capsize=2, capthick=1, linewidth=1)

    ax.set_xlabel('Level')
    ax.set_ylabel('Tie Rate')
    ax.set_title('Probabilistic: Tie Rate by Level\n[Error bars: 95% Wilson CI]')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.set_ylim(0, 1)
    _legend_outside(ax)

    fig.tight_layout()
    path = os.path.join(output_dir, 'prob_tie_by_level.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_pref_opp_by_level(runs: List[RunData], level_names: List[str], output_dir: str, mode: str):
    """Proper comparison: grouped bars per level with hue=run for both Pref and Opp (decided).
    
    Now includes 95% confidence interval error bars computed using Wilson score interval.
    """
    _ensure_matplotlib()
    _setup_style()

    n_runs = len(runs)
    x = np.arange(len(level_names))
    width = 0.8 / max(n_runs, 1)

    fig, (ax_pref, ax_opp) = plt.subplots(
        1, 2, figsize=(max(10, len(level_names) * 1.2) + 6, 5), sharey=True
    )

    for i, run in enumerate(runs):
        pref_rates = []
        opp_rates = []
        pref_errors_lower = []
        pref_errors_upper = []
        opp_errors_lower = []
        opp_errors_upper = []
        
        for level in level_names:
            level_data = run.summary.get('levels', {}).get(level, {})
            if mode == 'generation':
                pref, opp = _get_generation_pref_opp_rates(level_data)
                # Get counts for CI computation
                counts = get_level_counts(run.details, level, mode='generation')
            else:
                pref, opp = _get_prob_pref_opp_rates(level_data)
                counts = get_level_counts(run.details, level, mode='probabilistic')
            
            # Handle NaN values
            pref = pref if not math.isnan(pref) else 0.0
            opp = opp if not math.isnan(opp) else 0.0
            pref_rates.append(pref)
            opp_rates.append(opp)
            
            # Compute Wilson score confidence intervals
            pref_count, opp_count = counts
            total = pref_count + opp_count
            
            if total > 0:
                pref_ci = wilson_score_interval(pref_count, total)
                opp_ci = wilson_score_interval(opp_count, total)
                pref_errors_lower.append(max(0, pref - pref_ci[0]))
                pref_errors_upper.append(max(0, pref_ci[1] - pref))
                opp_errors_lower.append(max(0, opp - opp_ci[0]))
                opp_errors_upper.append(max(0, opp_ci[1] - opp))
            else:
                pref_errors_lower.append(0)
                pref_errors_upper.append(0)
                opp_errors_lower.append(0)
                opp_errors_upper.append(0)

        offset = (i - (n_runs - 1) / 2) * width
        
        # Plot bars with error bars
        ax_pref.bar(x + offset, pref_rates, width, label=run.label, color=run.color)
        ax_pref.errorbar(x + offset, pref_rates, 
                         yerr=[pref_errors_lower, pref_errors_upper],
                         fmt='none', color='black', capsize=2, capthick=1, linewidth=1)
        
        ax_opp.bar(x + offset, opp_rates, width, label=run.label, color=run.color)
        ax_opp.errorbar(x + offset, opp_rates,
                        yerr=[opp_errors_lower, opp_errors_upper],
                        fmt='none', color='black', capsize=2, capthick=1, linewidth=1)

    for ax in (ax_pref, ax_opp):
        ax.set_xticks(x)
        ax.set_xticklabels(level_names)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Level')

    ax_pref.set_ylabel('Rate (Decided)')
    ax_pref.set_title('Preference')
    ax_opp.set_title('Opposite')

    _legend_outside(ax_opp)

    title = 'Generation: Preference vs Opposite by Level (Decided)\n[Error bars: 95% Wilson CI]'
    filename = 'gen_pref_opp_by_level.png'
    if mode == 'probabilistic':
        title = 'Probabilistic: Preference vs Opposite by Level (Decided)\n[Error bars: 95% Wilson CI]'
        filename = 'prob_pref_opp_by_level.png'

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_pref_delta_by_level(runs: List[RunData], level_names: List[str], output_dir: str, mode: str):
    """Delta vs baseline (runs[0]) so increases/decreases are easy to spot.
    
    Uses paired t-test on per-question data for significance testing.
    This accounts for the fact that we're comparing the same questions across runs.
    """
    _ensure_matplotlib()
    _setup_style()

    if len(runs) < 2:
        return None

    base = runs[0]
    others = runs[1:]
    x = np.arange(len(level_names))
    n_other = len(others)
    width = 0.8 / max(n_other, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(level_names) * 1.2) + 3, 5.5))

    for i, run in enumerate(others):
        deltas = []
        delta_errors = []
        p_values = []
        test_info = []
        
        for level in level_names:
            # Get paired per-question data for paired t-test
            values_base, values_run, n_matched = get_paired_question_values(
                base.details, run.details, level, mode=mode
            )

            if n_matched >= 2:
                differences = [vr - vb for vb, vr in zip(values_base, values_run)]
                if HAS_NUMPY:
                    mean_diff = float(np.mean(differences))
                    std_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                else:
                    mean_diff = sum(differences) / len(differences)
                    if len(differences) > 1:
                        var_diff = sum((d - mean_diff) ** 2 for d in differences) / (len(differences) - 1)
                        std_diff = math.sqrt(var_diff)
                    else:
                        std_diff = 0.0

                deltas.append(mean_diff)

                if len(differences) > 1:
                    se_diff = std_diff / math.sqrt(len(differences))
                    if HAS_SCIPY:
                        t_crit = scipy_stats.t.ppf(0.975, df=len(differences) - 1)
                    else:
                        t_crit = 1.96
                    delta_errors.append(max(0, t_crit * se_diff))
                else:
                    delta_errors.append(0)

                p, test_name = paired_t_test(values_base, values_run)
                p_values.append(p)
                test_info.append(f"n={n_matched}, {test_name}")
            else:
                # Fallback to summary delta when insufficient paired data
                base_level = base.summary.get('levels', {}).get(level, {})
                run_level = run.summary.get('levels', {}).get(level, {})
                if mode == 'generation':
                    b = _get_generation_pref_rate(base_level)
                    r = _get_generation_pref_rate(run_level)
                else:
                    b = _get_prob_pref_rate(base_level)
                    r = _get_prob_pref_rate(run_level)
                b = b if not math.isnan(b) else 0.0
                r = r if not math.isnan(r) else 0.0
                deltas.append(r - b)

                # Fallback to propagated CI error when insufficient paired data
                base_counts = get_level_counts(base.details, level, mode=mode)
                run_counts = get_level_counts(run.details, level, mode=mode)
                base_pref, base_opp = base_counts
                run_pref, run_opp = run_counts
                base_total = base_pref + base_opp
                run_total = run_pref + run_opp
                if base_total > 0 and run_total > 0:
                    base_ci = wilson_score_interval(base_pref, base_total)
                    run_ci = wilson_score_interval(run_pref, run_total)
                    base_err = (base_ci[1] - base_ci[0]) / 2
                    run_err = (run_ci[1] - run_ci[0]) / 2
                    delta_errors.append(max(0, math.sqrt(base_err**2 + run_err**2)))
                else:
                    delta_errors.append(0)
                p_values.append(float('nan'))
                test_info.append(f"n={n_matched} (too few)")

        offset = (i - (n_other - 1) / 2) * width
        ax.bar(x + offset, deltas, width, label=f"{run.label} - {base.label}", color=run.color)
        ax.errorbar(x + offset, deltas, yerr=delta_errors,
                    fmt='none', color='black', capsize=3, capthick=1, linewidth=1)
        
        # Add significance stars
        for j, (delta, p) in enumerate(zip(deltas, p_values)):
            stars = significance_stars(p)
            if stars:
                y_pos = delta + delta_errors[j] + 0.02 if delta >= 0 else delta - delta_errors[j] - 0.02
                ax.text(x[j] + offset, y_pos, stars, ha='center', va='bottom' if delta >= 0 else 'top',
                        fontsize=10, fontweight='bold', color='red')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.set_xlabel('Level')
    if mode == 'generation':
        ax.set_ylabel('Δ Preference Rate (Decided)')
        mode_label = 'Generation'
        title = f'{mode_label}: Δ Preference vs Baseline by Level (Decided)\n[* p<0.05, ** p<0.01, *** p<0.001; Paired t-test]'
    else:
        ax.set_ylabel('Δ Margin (Preferred - Opposite)')
        mode_label = 'Probabilistic'
        title = f'{mode_label}: Δ Margin vs Baseline by Level\n[* p<0.05, ** p<0.01, *** p<0.001; Paired t-test]'
    ax.set_title(title)
    _legend_outside(ax)

    fig.tight_layout()
    filename = 'gen_pref_delta_by_level.png' if mode == 'generation' else 'prob_pref_delta_by_level.png'
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_pref_by_level_topic(runs: List[RunData], level_names: List[str], topics: List[str], output_dir: str, mode: str):
    """Heatmap showing preference rates by level and topic.
    
    Annotations include 95% CI in format: rate [CI_lower, CI_upper]
    """
    _ensure_matplotlib()
    _setup_style()

    n_runs = len(runs)
    fig_height = max(4.5, 0.4 * len(level_names) + 2)
    fig_width = max(6, 4.5 * n_runs)
    fig, axes = plt.subplots(
        1, n_runs, figsize=(fig_width, fig_height), sharey=True, constrained_layout=True
    )
    if n_runs == 1:
        axes = [axes]

    last_im = None
    for ax, run in zip(axes, runs):
        matrix = []
        ci_matrix = []  # Store (lower, upper) for each cell
        
        for level in level_names:
            level_data = run.summary.get('levels', {}).get(level, {})
            row = []
            ci_row = []
            
            for topic in topics:
                if mode == 'generation':
                    gen_per_topic = level_data.get('generation', {}).get('per_topic', {})
                    if _gen_topic_exists(gen_per_topic, topic):
                        rate = _get_generation_topic_pref_rate(gen_per_topic, topic)
                        counts = get_topic_counts(run.details, level, topic, mode='generation')
                    else:
                        rate = float('nan')
                        counts = (0, 0)
                else:
                    prob_per_topic = level_data.get('probabilistic', {}).get('per_topic', {})
                    if topic in prob_per_topic.get('counts', {}):
                        rate = _get_prob_topic_pref_rate(prob_per_topic, topic)
                        counts = get_topic_counts(run.details, level, topic, mode='probabilistic')
                    else:
                        rate = float('nan')
                        counts = (0, 0)
                
                row.append(rate)
                
                # Compute CI
                pref, opp = counts
                total = pref + opp
                if total > 0 and not np.isnan(rate):
                    ci = wilson_score_interval(pref, total)
                    ci_row.append(ci)
                else:
                    ci_row.append((float('nan'), float('nan')))
            
            matrix.append(row)
            ci_matrix.append(ci_row)

        data = np.array(matrix, dtype=float)
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        last_im = im

        ax.set_xticks(range(len(topics)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_title(run.label)

        # Annotate with rate and CI for small grids
        if len(topics) <= 12 and len(level_names) <= 8:
            for i in range(len(level_names)):
                for j in range(len(topics)):
                    val = data[i, j]
                    if not np.isnan(val):
                        color = 'white' if val > 0.5 else 'black'
                        ci = ci_matrix[i][j]
                        if not np.isnan(ci[0]):
                            # Show rate with CI width indicator
                            ci_width = ci[1] - ci[0]
                            text = f'{val:.2f}\n±{ci_width/2:.2f}'
                        else:
                            text = f'{val:.2f}'
                        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

    axes[0].set_ylabel('Level')

    title = 'Generation: Preference Rate by Level & Topic (Decided)\n[±95% CI half-width]'
    filename = 'gen_pref_by_level_topic.png'
    if mode == 'probabilistic':
        title = 'Probabilistic: Preference Rate by Level & Topic (Decided)\n[±95% CI half-width]'
        filename = 'prob_pref_by_level_topic.png'

    fig.suptitle(title, fontsize=12, fontweight='bold')
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04, label='Preference Rate')

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_pref_delta_by_level_topic(runs: List[RunData], level_names: List[str], topics: List[str], output_dir: str, mode: str):
    """Delta heatmaps vs baseline (runs[0]) for proper run-to-run comparison.
    
    Uses paired t-test on per-question data for significance testing.
    """
    _ensure_matplotlib()
    _setup_style()

    if len(runs) < 2:
        return []

    base = runs[0]
    out_paths = []
    mode_label = 'Generation' if mode == 'generation' else 'Probabilistic'

    for run in runs[1:]:
        matrix = []
        p_values = []  # Store p-values for significance markers
        
        for level in level_names:
            base_level = base.summary.get('levels', {}).get(level, {})
            run_level = run.summary.get('levels', {}).get(level, {})

            row = []
            p_row = []
            for topic in topics:
                # Paired per-question values for this topic
                values_base, values_run, n_matched = get_paired_question_values(
                    base.details, run.details, level, mode=mode, topic=topic
                )
                if n_matched >= 2:
                    differences = [vr - vb for vb, vr in zip(values_base, values_run)]
                    if HAS_NUMPY:
                        delta = float(np.mean(differences))
                    else:
                        delta = sum(differences) / len(differences)
                    p, _ = paired_t_test(values_base, values_run)
                else:
                    delta = float('nan')
                    p = float('nan')
                
                row.append(delta)
                p_row.append(p)
            
            matrix.append(row)
            p_values.append(p_row)

        data = np.array(matrix, dtype=float)
        masked = np.ma.masked_invalid(data)

        fig_height = max(4.5, 0.4 * len(level_names) + 2)
        fig_width = max(7, 0.55 * len(topics) + 3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if HAS_NUMPY:
            max_abs = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 0.5
        else:
            max_abs = 0.5
        limit = max(0.5, max_abs)
        im = ax.imshow(masked, cmap='RdBu_r', aspect='auto', vmin=-limit, vmax=limit)

        ax.set_xticks(range(len(topics)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Level')
        if mode == 'generation':
            title_metric = "Δ Pref (Decided)"
        else:
            title_metric = "Δ Margin"
        ax.set_title(f"{mode_label}: {title_metric}: {run.label} - {base.label}\n[* p<0.05, ** p<0.01, *** p<0.001; Paired t-test]")

        # Annotate every cell with the delta value and significance markers
        n_cells = len(level_names) * len(topics)
        if n_cells > 500:
            fs = 4
        elif n_cells > 300:
            fs = 5
        elif n_cells > 150:
            fs = 6
        elif n_cells > 80:
            fs = 7
        else:
            fs = 8

        for i in range(len(level_names)):
            for j in range(len(topics)):
                val = data[i, j]
                if np.isnan(val):
                    continue
                color = 'white' if abs(val) > 0.25 else 'black'
                stars = significance_stars(p_values[i][j])
                text = f'{val:+.2f}{stars}'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=fs)

        fig.colorbar(im, ax=ax, label='Δ Preference Rate')

        fig.tight_layout()
        prefix = 'gen' if mode == 'generation' else 'prob'
        filename = f"{prefix}_pref_delta_by_level_topic_{_slugify(run.label)}_vs_{_slugify(base.label)}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        out_paths.append(path)

    return out_paths


def _default_output_dir(labels: List[str], summary_paths: List[str]) -> str:
    base = Path('outputs') / 'eval' / 'comparisons'
    suffix = '_'.join(_slugify(l) for l in labels)
    return str(base / f"compare_{suffix}")


def main():
    parser = argparse.ArgumentParser(description='Compare 2-3 evaluation summary runs with statistical significance')
    parser.add_argument('summaries', nargs='+', help='summary.json paths or RUN_IDs (2 or 3)')
    parser.add_argument('--output-dir', '-o', help='Output directory for charts')
    parser.add_argument('--labels', nargs='+', help='Human-friendly run names (2 or 3), same order as summaries')
    parser.add_argument('--no-stats', action='store_true', help='Disable statistical significance features')

    args = parser.parse_args()
    if len(args.summaries) < 2 or len(args.summaries) > 3:
        raise SystemExit('Please provide 2 or 3 summaries (paths or RUN_IDs).')

    summaries = []
    labels = []
    paths = []
    details_list = []

    for arg in args.summaries:
        path = resolve_summary_path(arg)
        summary = load_summary(path)
        run_id = str(summary.get('run_id', '')).strip()
        run_label = str(summary.get('run_label', '')).strip()
        if run_label:
            label = run_label
        else:
            label = run_id if run_id else Path(path).parent.name
        summaries.append(summary)
        labels.append(label)
        paths.append(path)
        
        # Load per-question details for statistical significance
        if not args.no_stats:
            details = load_details_from_shards(summary)
            details_list.append(details)
        else:
            details_list.append({})

    if args.labels is not None:
        if len(args.labels) != len(labels):
            raise SystemExit(f"--labels must have {len(labels)} values (got {len(args.labels)})")
        labels = args.labels

    output_dir = args.output_dir or _default_output_dir(labels, paths)
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#3498db', '#9b59b6', '#f39c12']
    runs = [RunData(label=label, path=path, summary=summary, color=colors[i % len(colors)], details=details)
            for i, (label, path, summary, details) in enumerate(zip(labels, paths, summaries, details_list))]

    level_names = _get_level_names(runs)
    gen_topics = _collect_generation_topics(runs)
    prob_topics = _collect_prob_topics(runs)

    print(f"Generating comparison charts in: {output_dir}")
    print(f"Runs: {', '.join(labels)}")
    for run in runs:
        run_id = str(run.summary.get('run_id', 'N/A'))
        n_details = sum(len(v) for v in run.details.values())
        print(f"  - {run.label}: run_id={run_id}, summary={run.path}, details={n_details} questions")
    
    if HAS_SCIPY:
        print("Statistical significance: enabled (scipy available)")
    else:
        print("Statistical significance: limited (scipy not available, install for chi-square tests)")

    # Generation comparisons
    if HAS_MATPLOTLIB and HAS_NUMPY:
        plot_refusal_by_level(runs, level_names, output_dir)
        plot_pref_opp_by_level(runs, level_names, output_dir, mode='generation')
        plot_pref_delta_by_level(runs, level_names, output_dir, mode='generation')
        if gen_topics:
            plot_pref_by_level_topic(runs, level_names, gen_topics, output_dir, mode='generation')
            plot_pref_delta_by_level_topic(runs, level_names, gen_topics, output_dir, mode='generation')
        else:
            print("[!] No generation per-topic data found; skipping gen topic heatmap")

        # Probabilistic comparisons
        plot_prob_tie_by_level(runs, level_names, output_dir)
        plot_pref_opp_by_level(runs, level_names, output_dir, mode='probabilistic')
        plot_pref_delta_by_level(runs, level_names, output_dir, mode='probabilistic')
        if prob_topics:
            plot_pref_by_level_topic(runs, level_names, prob_topics, output_dir, mode='probabilistic')
            plot_pref_delta_by_level_topic(runs, level_names, prob_topics, output_dir, mode='probabilistic')
        else:
            print("[!] No probabilistic per-topic data found; skipping prob topic heatmap")
        
        # Generate summary statistics report
        _generate_stats_report(runs, level_names, gen_topics, prob_topics, output_dir)
    else:
        print("[!] matplotlib/numpy not available; no charts generated")

    print("✓ Comparison complete!")


def _generate_stats_report(runs: List[RunData], level_names: List[str], 
                           gen_topics: List[str], prob_topics: List[str], output_dir: str):
    """Generate a text report with statistical test results using paired t-tests."""
    report_lines = [
        "=" * 70,
        "STATISTICAL SIGNIFICANCE REPORT",
        "=" * 70,
        "",
        f"Runs compared: {', '.join(r.label for r in runs)}",
        f"Levels: {', '.join(level_names)}",
        f"Generation topics: {', '.join(gen_topics) if gen_topics else 'N/A'}",
        f"Probabilistic topics: {', '.join(prob_topics) if prob_topics else 'N/A'}",
        "",
        "Test: Paired t-test on per-question data",
        "(Compares the same questions across runs, accounting for question-level variance)",
        "",
    ]
    
    if len(runs) < 2:
        report_lines.append("Need at least 2 runs for comparison.")
    else:
        base = runs[0]
        
        for run in runs[1:]:
            report_lines.extend([
                "-" * 70,
                f"Comparison: {run.label} vs {base.label} (baseline)",
                "-" * 70,
                "",
            ])
            
            # Level-wise comparison for generation
            report_lines.append("GENERATION EVALUATION (by Level):")
            report_lines.append("-" * 40)
            
            for level in level_names:
                values_base, values_run, n_matched = get_paired_question_values(
                    base.details, run.details, level, mode='generation'
                )
                if n_matched > 0:
                    if HAS_NUMPY:
                        base_rate = float(np.mean(values_base))
                        run_rate = float(np.mean(values_run))
                    else:
                        base_rate = sum(values_base) / len(values_base)
                        run_rate = sum(values_run) / len(values_run)
                    delta = run_rate - base_rate
                else:
                    base_rate = float('nan')
                    run_rate = float('nan')
                    delta = float('nan')

                base_pref, base_opp = get_level_counts(base.details, level, mode='generation')
                run_pref, run_opp = get_level_counts(run.details, level, mode='generation')
                base_total = base_pref + base_opp
                run_total = run_pref + run_opp
                
                # Paired t-test
                values_base, values_run, n_matched = get_paired_question_values(
                    base.details, run.details, level, mode='generation'
                )
                p, test_name = paired_t_test(values_base, values_run)
                stars = significance_stars(p)
                
                p_str = f"{p:.4f}" if not math.isnan(p) else "N/A"
                
                report_lines.append(
                    f"  {level}: {base.label}={base_rate:.3f} ({base_pref}/{base_total}), "
                    f"{run.label}={run_rate:.3f} ({run_pref}/{run_total}), "
                    f"Δ={delta:+.3f}, n={n_matched}, p={p_str} {stars}"
                )
            
            report_lines.append("")
            
            # Level-wise comparison for probabilistic (using margins)
            report_lines.append("PROBABILISTIC EVALUATION (by Level):")
            report_lines.append("-" * 40)
            report_lines.append("  (Test on log-prob margins; positive margin = preference wins)")
            
            for level in level_names:
                values_base, values_run, n_matched = get_paired_question_values(
                    base.details, run.details, level, mode='probabilistic'
                )
                if n_matched > 0:
                    if HAS_NUMPY:
                        base_rate = float(np.mean(values_base))
                        run_rate = float(np.mean(values_run))
                    else:
                        base_rate = sum(values_base) / len(values_base)
                        run_rate = sum(values_run) / len(values_run)
                    delta = run_rate - base_rate
                else:
                    base_rate = float('nan')
                    run_rate = float('nan')
                    delta = float('nan')
                
                # Paired t-test on margins
                values_base, values_run, n_matched = get_paired_question_values(
                    base.details, run.details, level, mode='probabilistic'
                )
                p, test_name = paired_t_test(values_base, values_run)
                stars = significance_stars(p)
                
                p_str = f"{p:.4f}" if not math.isnan(p) else "N/A"
                
                report_lines.append(
                    f"  {level}: {base.label}={base_rate:.3f}, {run.label}={run_rate:.3f}, "
                    f"Δ={delta:+.3f}, n={n_matched}, p={p_str} {stars}"
                )
            
            report_lines.append("")
    
    report_lines.extend([
        "",
        "=" * 70,
        "Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
        "Test: Paired t-test (mean-based)",
        "  - For generation: compares pref_rate_decided per question",
        "  - For probabilistic: compares log-prob margins per question",
        "Confidence intervals: Wilson score interval (95%)",
        "=" * 70,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(output_dir, "statistical_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n{report_text}\n")
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
