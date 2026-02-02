#!/usr/bin/env python3
"""
Visualize evaluation summary results with tables and charts.

Usage:
    python visualize_eval_summary.py outputs/eval/eval_123_merged/summary.json
    python visualize_eval_summary.py outputs/eval/eval_123_merged/summary.json --output-dir reports/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def load_summary(path: str) -> dict:
    """Load summary JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f" {title}".center(width))
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 60):
    """Print a formatted subheader."""
    print()
    print(f"  {title}")
    print(f"  {char * len(title)}")


def format_percent(value: float) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.1f}%"


def format_margin(value: float) -> str:
    """Format margin value with sign."""
    return f"{value:+.4f}"


def print_simple_table(headers: list, rows: list, indent: int = 4):
    """Print a simple ASCII table without tabulate."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    indent_str = " " * indent
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    
    print(f"{indent_str}{header_line}")
    print(f"{indent_str}{separator}")
    
    # Print rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(f"{indent_str}{row_line}")


def print_table(headers: list, rows: list, tablefmt: str = "simple", indent: int = 4):
    """Print a formatted table."""
    if HAS_TABULATE:
        table = tabulate(rows, headers=headers, tablefmt=tablefmt)
        for line in table.split('\n'):
            print(" " * indent + line)
    else:
        print_simple_table(headers, rows, indent)


def print_config_summary(config: dict):
    """Print configuration summary."""
    print_header("CONFIGURATION")
    
    # Model info
    print_subheader("Model")
    model = config.get('model', {})
    target = model.get('target', 'N/A')
    # Shorten long paths
    if len(target) > 60:
        target = "..." + target[-57:]
    print(f"    Target: {target}")
    print(f"    Judge:  {model.get('judge', 'N/A')}")
    
    # Data info
    print_subheader("Data")
    data = config.get('data', {})
    print(f"    Topics: {', '.join(data.get('topic_ids', []))}")
    levels = [l['name'] for l in data.get('levels', []) if l.get('enabled', True)]
    print(f"    Levels: {', '.join(levels)}")
    
    # Generation settings
    print_subheader("Generation Settings")
    gen = config.get('generation', {})
    print(f"    Samples: {gen.get('num_samples', 'N/A')}")
    print(f"    Temperature: {gen.get('temperature', 'N/A')}")
    print(f"    Max tokens: {gen.get('max_new_tokens', 'N/A')}")


def print_overall_summary(levels: dict):
    """Print overall summary table across all levels."""
    print_header("OVERALL RESULTS SUMMARY")
    
    # Generation results - show both rate perspectives
    print_subheader("Generation-based Evaluation (Judge) - Response Rates")
    headers = ["Level", "N", "Pref/All", "Opp/All", "Refusal", "Pref/Dec", "Opp/Dec"]
    rows = []
    
    all_pref_rates_all = []
    all_opp_rates_all = []
    all_refusal_rates = []
    all_pref_rates_decided = []
    all_opp_rates_decided = []
    total_n = 0
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        gen = level_data.get('generation', {})
        n = gen.get('total_questions', level_data.get('num_questions', 0))
        
        # Compute rates from response_counts (always available)
        counts = gen.get('response_counts', {})
        pref = counts.get('preference', 0)
        opp = counts.get('opposite', 0)
        unk = counts.get('unknown', 0)
        total_all = pref + opp + unk
        total_dec = pref + opp
        
        pref_rate_all = pref / total_all if total_all > 0 else 0
        opp_rate_all = opp / total_all if total_all > 0 else 0
        refusal_rate = unk / total_all if total_all > 0 else 0
        pref_rate_decided = pref / total_dec if total_dec > 0 else 0.5
        opp_rate_decided = opp / total_dec if total_dec > 0 else 0.5
        
        all_pref_rates_all.append((pref_rate_all, n))
        all_opp_rates_all.append((opp_rate_all, n))
        all_refusal_rates.append((refusal_rate, n))
        all_pref_rates_decided.append((pref_rate_decided, n))
        all_opp_rates_decided.append((opp_rate_decided, n))
        total_n += n
        
        rows.append([
            level_name,
            n,
            format_percent(pref_rate_all),
            format_percent(opp_rate_all),
            format_percent(refusal_rate),
            format_percent(pref_rate_decided),
            format_percent(opp_rate_decided)
        ])
    
    # Add weighted average total row
    if total_n > 0:
        avg_pref_all = sum(r * n for r, n in all_pref_rates_all) / total_n
        avg_opp_all = sum(r * n for r, n in all_opp_rates_all) / total_n
        avg_refusal = sum(r * n for r, n in all_refusal_rates) / total_n
        avg_pref_dec = sum(r * n for r, n in all_pref_rates_decided) / total_n
        avg_opp_dec = sum(r * n for r, n in all_opp_rates_decided) / total_n
        rows.append([
            "AVG",
            total_n,
            format_percent(avg_pref_all),
            format_percent(avg_opp_all),
            format_percent(avg_refusal),
            format_percent(avg_pref_dec),
            format_percent(avg_opp_dec)
        ])
    
    print_table(headers, rows, "simple")
    
    # Probabilistic results
    print_subheader("Probabilistic Evaluation (Log-prob)")
    headers = ["Level", "N", "Preference", "Opposite", "Tie", "Mean Margin"]
    rows = []
    
    total_pref = 0
    total_opp = 0
    total_tie = 0
    total_n = 0
    margin_sum = 0
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        prob = level_data.get('probabilistic', {})
        rates = prob.get('rates', {})
        counts = prob.get('counts', {})
        n = prob.get('total_scored', 0)
        margin = prob.get('mean_margin', 0)
        
        total_pref += counts.get('preference', 0)
        total_opp += counts.get('opposite', 0)
        total_tie += counts.get('tie', 0)
        total_n += n
        margin_sum += margin * n
        
        rows.append([
            level_name,
            n,
            format_percent(rates.get('preference', 0)),
            format_percent(rates.get('opposite', 0)),
            format_percent(rates.get('tie', 0)),
            format_margin(margin)
        ])
    
    # Add total row
    if total_n > 0:
        rows.append([
            "TOTAL",
            total_n,
            format_percent(total_pref / total_n),
            format_percent(total_opp / total_n),
            format_percent(total_tie / total_n),
            format_margin(margin_sum / total_n)
        ])
    
    print_table(headers, rows, "simple")


def print_per_topic_breakdown(levels: dict):
    """Print per-topic breakdown for each level."""
    print_header("PER-TOPIC BREAKDOWN")
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        
        print_subheader(f"Level {level_name}")
        
        # Generation per topic - use new metrics format
        gen_per_topic = level_data.get('generation', {}).get('per_topic', {})
        # Extract topics using helper
        topics = _extract_generation_topics(gen_per_topic)
        if topics:
            print()
            print("    Generation (Judge) - Response Rates:")
            headers = ["Topic", "Pref/All", "Opp/All", "Refusal", "Pref/Dec", "Opp/Dec"]
            rows = []
            
            for topic in sorted(topics):
                counts = _get_generation_topic_counts(gen_per_topic, topic)
                pref = counts.get('preference', 0)
                opp = counts.get('opposite', 0)
                unk = counts.get('unknown', 0)
                total = pref + opp + unk
                decided = pref + opp
                
                pref_all = pref / total if total > 0 else 0
                opp_all = opp / total if total > 0 else 0
                refusal = unk / total if total > 0 else 0
                pref_dec = pref / decided if decided > 0 else 0.5
                opp_dec = opp / decided if decided > 0 else 0.5
                
                rows.append([
                    topic,
                    format_percent(pref_all),
                    format_percent(opp_all),
                    format_percent(refusal),
                    format_percent(pref_dec),
                    format_percent(opp_dec)
                ])
            
            print_table(headers, rows, "simple", indent=6)
        
        # Probabilistic per topic
        prob_per_topic = level_data.get('probabilistic', {}).get('per_topic', {})
        if prob_per_topic:
            print()
            print("    Probabilistic (Log-prob):")
            headers = ["Topic", "Pref", "Opp", "Tie", "Pref Rate", "Margin"]
            rows = []
            
            topic_counts = prob_per_topic.get('counts', {})
            topic_margins = prob_per_topic.get('mean_margins', {})
            
            for topic in sorted(topic_counts.keys()):
                counts = topic_counts[topic]
                # Only count preference vs opposite (exclude tie)
                decided = counts.get('preference', 0) + counts.get('opposite', 0)
                pref_rate = counts.get('preference', 0) / decided if decided > 0 else 0
                margin = topic_margins.get(topic, 0)
                rows.append([
                    topic,
                    counts.get('preference', 0),
                    counts.get('opposite', 0),
                    counts.get('tie', 0),
                    format_percent(pref_rate),
                    format_margin(margin)
                ])
            
            print_table(headers, rows, "simple", indent=6)


def print_visual_bars(levels: dict, width: int = 40):
    """Print ASCII visual bars for quick understanding."""
    print_header("VISUAL SUMMARY (Generation - Decided)")
    
    print()
    print("    Preference rate (decided only) per level")
    print("    " + "─" * (width + 20))
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        gen = level_data.get('generation', {})
        # Use new averaged metric, fallback to old format
        pref_rate = gen.get('mean_pref_rate_decided', gen.get('response_rates', {}).get('preference', 0))
        
        filled = int(pref_rate * width)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        
        # Color coding via emoji
        if pref_rate >= 0.7:
            indicator = "🟢"
        elif pref_rate >= 0.4:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"    {level_name}: [{bar}] {format_percent(pref_rate)} {indicator}")
    
    print()
    print_header("VISUAL SUMMARY (Probabilistic)")
    print()
    print("    Preference rate per level (higher = model aligns with preference)")
    print("    " + "─" * (width + 20))
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        prob = level_data.get('probabilistic', {})
        pref_rate = prob.get('rates', {}).get('preference', 0)
        
        filled = int(pref_rate * width)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        
        # Color coding via emoji
        if pref_rate >= 0.7:
            indicator = "🟢"
        elif pref_rate >= 0.4:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"    {level_name}: [{bar}] {format_percent(pref_rate)} {indicator}")


def _get_generation_response_counts(levels: dict, level_names: list):
    """Extract generation response counts with fallback for different formats."""
    pref_counts = []
    opp_counts = []
    unk_counts = []
    
    for l in level_names:
        gen = levels[l].get('generation', {})
        counts = gen.get('response_counts', {})
        pref_counts.append(counts.get('preference', 0))
        opp_counts.append(counts.get('opposite', 0))
        unk_counts.append(counts.get('unknown', 0))
    
    return pref_counts, opp_counts, unk_counts


def _extract_generation_topics(gen_per_topic: dict) -> set:
    """Extract actual topic IDs from generation per_topic data.
    
    Handles two formats:
    1. New format: per_topic = {topic_id: {response_counts: {...}, ...}, ...}
    2. Old/merged format: per_topic = {response_counts: {topic_id: {...}}, ...}
    """
    topics = set()
    if not gen_per_topic:
        return topics
    
    # Check for old/merged format
    if 'response_counts' in gen_per_topic and isinstance(gen_per_topic.get('response_counts'), dict):
        topics.update(gen_per_topic['response_counts'].keys())
    else:
        # New format: top-level keys are topic IDs (skip known metadata keys)
        skip_keys = {'response_counts', 'question_majority_counts', 'mean_pref_rate_all', 
                     'mean_pref_rate_decided', 'mean_refusal_rate'}
        for key in gen_per_topic.keys():
            if key not in skip_keys and isinstance(gen_per_topic[key], dict):
                topics.add(key)
    
    return topics


def _get_generation_topic_counts(gen_per_topic: dict, topic: str) -> dict:
    """Get response counts for a topic from generation data.
    
    Handles two formats:
    1. New format: per_topic = {topic_id: {response_counts: {...}, ...}, ...}
    2. Old/merged format: per_topic = {response_counts: {topic_id: {...}}, ...}
    """
    if not gen_per_topic:
        return {}
    
    # Check for old/merged format
    if 'response_counts' in gen_per_topic and isinstance(gen_per_topic.get('response_counts'), dict):
        return gen_per_topic['response_counts'].get(topic, {})
    else:
        # New format
        topic_data = gen_per_topic.get(topic, {})
        if isinstance(topic_data, dict):
            return topic_data.get('response_counts', topic_data)
    
    return {}


def _get_generation_topic_pref_rate(gen_per_topic: dict, topic: str, rate_type: str = 'decided'):
    """Get preference rate for a topic from generation data."""
    # Check for old/merged format first
    if 'response_counts' in gen_per_topic and isinstance(gen_per_topic.get('response_counts'), dict):
        counts = gen_per_topic['response_counts'].get(topic, {})
    else:
        # New format
        topic_data = gen_per_topic.get(topic, {})
        if isinstance(topic_data, dict):
            # New format has mean rates directly
            if rate_type == 'decided' and 'mean_pref_rate_decided' in topic_data:
                return topic_data.get('mean_pref_rate_decided', 0.5)
            elif rate_type == 'all' and 'mean_pref_rate_all' in topic_data:
                return topic_data.get('mean_pref_rate_all', 0.0)
            # Fallback: compute from response_counts
            counts = topic_data.get('response_counts', topic_data)
        else:
            counts = {}
    
    # Compute rate from counts
    pref = counts.get('preference', 0)
    opp = counts.get('opposite', 0)
    unk = counts.get('unknown', 0)
    
    if rate_type == 'decided':
        decided = pref + opp
        return pref / decided if decided > 0 else 0.5
    else:  # all
        total = pref + opp + unk
        return pref / total if total > 0 else 0.0


def create_matplotlib_charts(summary: dict, output_dir: str):
    """Create matplotlib charts and save them."""
    if not HAS_MATPLOTLIB:
        print("\n    [!] matplotlib not available, skipping chart generation")
        return
    
    levels = summary.get('levels', {})
    level_names = sorted(levels.keys())
    
    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
    
    # Larger figure: 3 rows x 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    run_label = summary.get('run_label') or summary.get('run_id', 'N/A')
    run_label = summary.get('run_label') or summary.get('run_id', 'N/A')
    run_id = summary.get('run_id', 'N/A')
    title_run = run_label if run_label == run_id else f"{run_label} (id: {run_id})"
    title_run = run_label if run_label == run_id else f"{run_label} (id: {run_id})"
    fig.suptitle(f"Evaluation Summary - {title_run}", fontsize=14, fontweight='bold')
    
    colors = {
        'preference': '#2ecc71',  # Green
        'opposite': '#e74c3c',    # Red
        'unknown': '#95a5a6',     # Gray
        'tie': '#f39c12'          # Orange
    }
    
    x = range(len(level_names))
    width = 0.25
    
    # ========== ROW 0: Generation Bar Charts ==========
    
    # [0,0] Generation response counts (Pref/Opp/Unknown) - like before
    ax = axes[0, 0]
    pref_counts, opp_counts, unk_counts = _get_generation_response_counts(levels, level_names)
    totals = [p + o + u for p, o, u in zip(pref_counts, opp_counts, unk_counts)]
    pref_rates = [p / t if t > 0 else 0 for p, t in zip(pref_counts, totals)]
    opp_rates = [o / t if t > 0 else 0 for o, t in zip(opp_counts, totals)]
    unk_rates = [u / t if t > 0 else 0 for u, t in zip(unk_counts, totals)]
    
    ax.bar([i - width for i in x], pref_rates, width, label='Preference', color=colors['preference'])
    ax.bar(x, opp_rates, width, label='Opposite', color=colors['opposite'])
    ax.bar([i + width for i in x], unk_rates, width, label='Refusal', color=colors['unknown'])
    ax.set_xlabel('Level')
    ax.set_ylabel('Rate')
    ax.set_title('Generation: Response Counts (All)')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # [0,1] Generation: Pref vs Opp rates (All denominator vs Decided denominator)
    ax = axes[0, 1]
    pref_rates_all = []
    opp_rates_all = []
    pref_rates_dec = []
    opp_rates_dec = []
    for l in level_names:
        gen = levels[l].get('generation', {})
        counts = gen.get('response_counts', {})
        pref = counts.get('preference', 0)
        opp = counts.get('opposite', 0)
        unk = counts.get('unknown', 0)
        total_all = pref + opp + unk
        total_dec = pref + opp
        
        pref_rates_all.append(pref / total_all if total_all > 0 else 0)
        opp_rates_all.append(opp / total_all if total_all > 0 else 0)
        pref_rates_dec.append(pref / total_dec if total_dec > 0 else 0.5)
        opp_rates_dec.append(opp / total_dec if total_dec > 0 else 0.5)
    
    # 4 bars per level: pref/all, opp/all, pref/decided, opp/decided
    width4 = 0.2
    ax.bar([i - 1.5*width4 for i in x], pref_rates_all, width4, label='Pref/All', color=colors['preference'], alpha=0.6)
    ax.bar([i - 0.5*width4 for i in x], opp_rates_all, width4, label='Opp/All', color=colors['opposite'], alpha=0.6)
    ax.bar([i + 0.5*width4 for i in x], pref_rates_dec, width4, label='Pref/Decided', color=colors['preference'])
    ax.bar([i + 1.5*width4 for i in x], opp_rates_dec, width4, label='Opp/Decided', color=colors['opposite'])
    ax.set_xlabel('Level')
    ax.set_ylabel('Rate')
    ax.set_title('Generation: Pref vs Opp (All vs Decided)')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    
    # [0,2] Probabilistic rates bar chart
    ax = axes[0, 2]
    pref_rates = [levels[l].get('probabilistic', {}).get('rates', {}).get('preference', 0) for l in level_names]
    opp_rates = [levels[l].get('probabilistic', {}).get('rates', {}).get('opposite', 0) for l in level_names]
    tie_rates = [levels[l].get('probabilistic', {}).get('rates', {}).get('tie', 0) for l in level_names]
    
    ax.bar([i - width for i in x], pref_rates, width, label='Preference', color=colors['preference'])
    ax.bar(x, opp_rates, width, label='Opposite', color=colors['opposite'])
    ax.bar([i + width for i in x], tie_rates, width, label='Tie', color=colors['tie'])
    ax.set_xlabel('Level')
    ax.set_ylabel('Rate')
    ax.set_title('Probabilistic: Results')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # ========== ROW 1: Generation Heatmaps ==========
    
    # Collect all topics across levels for generation
    all_topics_gen = set()
    for level_name in level_names:
        gen_per_topic = levels[level_name].get('generation', {}).get('per_topic', {})
        all_topics_gen.update(_extract_generation_topics(gen_per_topic))
    topics_gen = sorted(all_topics_gen)
    
    # [1,0] Generation heatmap - Pref Rate (All)
    ax = axes[1, 0]
    if topics_gen:
        heatmap_data = []
        for level_name in level_names:
            gen_per_topic = levels[level_name].get('generation', {}).get('per_topic', {})
            row = [_get_generation_topic_pref_rate(gen_per_topic, t, 'all') for t in topics_gen]
            heatmap_data.append(row)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(topics_gen)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics_gen)
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Level')
        ax.set_title('Generation: Pref Rate (All)')
        for i in range(len(level_names)):
            for j in range(len(topics_gen)):
                val = heatmap_data[i][j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No per-topic data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Generation: Pref Rate (All)')
    
    # [1,1] Generation heatmap - Pref Rate (Decided)
    ax = axes[1, 1]
    if topics_gen:
        heatmap_data = []
        for level_name in level_names:
            gen_per_topic = levels[level_name].get('generation', {}).get('per_topic', {})
            row = [_get_generation_topic_pref_rate(gen_per_topic, t, 'decided') for t in topics_gen]
            heatmap_data.append(row)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(topics_gen)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics_gen)
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Level')
        ax.set_title('Generation: Pref Rate (Decided)')
        for i in range(len(level_names)):
            for j in range(len(topics_gen)):
                val = heatmap_data[i][j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No per-topic data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Generation: Pref Rate (Decided)')
    
    # [1,2] Mean margin by level (probabilistic)
    ax = axes[1, 2]
    margins = [levels[l].get('probabilistic', {}).get('mean_margin', 0) for l in level_names]
    bar_colors = [colors['preference'] if m >= 0 else colors['opposite'] for m in margins]
    ax.bar(level_names, margins, color=bar_colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Level')
    ax.set_ylabel('Mean Margin')
    ax.set_title('Probabilistic: Mean Margin by Level')
    
    # ========== ROW 2: Probabilistic Heatmaps ==========
    
    # [2,0] Probabilistic heatmap - margins
    ax = axes[2, 0]
    
    # Collect all topics across levels for probabilistic
    all_topics_prob = set()
    for level_name in level_names:
        prob_per_topic = levels[level_name].get('probabilistic', {}).get('per_topic', {})
        topic_margins = prob_per_topic.get('mean_margins', {})
        all_topics_prob.update(topic_margins.keys())
    topics_prob = sorted(all_topics_prob)
    
    if topics_prob:
        heatmap_data = []
        for level_name in level_names:
            prob_per_topic = levels[level_name].get('probabilistic', {}).get('per_topic', {})
            topic_margins = prob_per_topic.get('mean_margins', {})
            row = [topic_margins.get(t, 0) for t in topics_prob]
            heatmap_data.append(row)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(topics_prob)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics_prob)
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Level')
        ax.set_title('Probabilistic: Mean Margin Heatmap')
        for i in range(len(level_names)):
            for j in range(len(topics_prob)):
                val = heatmap_data[i][j]
                color = 'white' if abs(val) > 0.25 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No per-topic data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Probabilistic: Mean Margin Heatmap')
    
    # [2,1] Probabilistic heatmap - preference rate
    ax = axes[2, 1]
    
    # Collect all topics
    all_topics_prob_counts = set()
    for level_name in level_names:
        prob_per_topic = levels[level_name].get('probabilistic', {}).get('per_topic', {})
        topic_counts = prob_per_topic.get('counts', {})
        all_topics_prob_counts.update(topic_counts.keys())
    topics_prob_counts = sorted(all_topics_prob_counts)
    
    if topics_prob_counts:
        heatmap_data = []
        for level_name in level_names:
            prob_per_topic = levels[level_name].get('probabilistic', {}).get('per_topic', {})
            topic_counts = prob_per_topic.get('counts', {})
            row = []
            for topic in topics_prob_counts:
                counts = topic_counts.get(topic, {})
                # Only count preference vs opposite (exclude tie)
                decided = counts.get('preference', 0) + counts.get('opposite', 0)
                pref_rate = counts.get('preference', 0) / decided if decided > 0 else 0.5
                row.append(pref_rate)
            heatmap_data.append(row)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(topics_prob_counts)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics_prob_counts)
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Level')
        ax.set_title('Probabilistic: Pref Rate Heatmap')
        
        # Add text annotations
        for i in range(len(level_names)):
            for j in range(len(topics_prob_counts)):
                val = heatmap_data[i][j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Preference Rate')
    else:
        ax.text(0.5, 0.5, 'No per-topic data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Probabilistic: Pref Rate Heatmap')
    
    # [2,2] Summary stats text
    ax = axes[2, 2]
    ax.axis('off')
    
    # Build summary text
    summary_lines = []
    summary_lines.append("Summary Statistics")
    summary_lines.append("-" * 30)
    for level_name in level_names:
        gen = levels[level_name].get('generation', {})
        prob = levels[level_name].get('probabilistic', {})
        
        gen_counts = gen.get('response_counts', {})
        gen_total = sum(gen_counts.values()) if gen_counts else 0
        gen_decided = gen_counts.get('preference', 0) + gen_counts.get('opposite', 0)
        
        prob_counts = prob.get('counts', {})
        prob_total = sum(prob_counts.values()) if prob_counts else 0
        
        summary_lines.append(f"\n{level_name}:")
        summary_lines.append(f"  Gen: {gen_total} total, {gen_decided} decided")
        summary_lines.append(f"  Prob: {prob_total} total")
    
    ax.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title('Summary')
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, 'eval_summary_charts.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n    📊 Charts saved to: {chart_path}")
    plt.close()


def generate_html_report(summary: dict, output_dir: str):
    """Generate an HTML report."""
    levels = summary.get('levels', {})
    config = summary.get('config', {})
    run_id = summary.get('run_id', 'N/A')
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Eval Summary - {title_run}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .bad {{ color: #e74c3c; font-weight: bold; }}
        .neutral {{ color: #f39c12; font-weight: bold; }}
        .config-box {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .progress-bar {{
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s;
        }}
        .pref {{ background: #27ae60; }}
        .opp {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <h1>📊 Evaluation Summary - {title_run}</h1>
    
    <div class="config-box">
        <h3>Configuration</h3>
        <p><strong>Run Label:</strong> <code>{run_label}</code></p>
        <p><strong>Run ID:</strong> <code>{run_id}</code></p>
        <p><strong>Target Model:</strong> <code>{config.get('model', {}).get('target', 'N/A')}</code></p>
        <p><strong>Judge Model:</strong> <code>{config.get('model', {}).get('judge', 'N/A')}</code></p>
        <p><strong>Topics:</strong> {', '.join(config.get('data', {}).get('topic_ids', []))}</p>
    </div>
    
    <h2>Overall Results</h2>
    
    <h3>Generation (Judge-based)</h3>
    <table>
        <tr>
            <th>Level</th>
            <th>Questions</th>
            <th>Preference</th>
            <th>Opposite</th>
            <th>Unknown</th>
            <th>Visual</th>
        </tr>
"""
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        gen = level_data.get('generation', {})
        rates = gen.get('response_rates', {})
        n = level_data.get('num_questions', 0)
        pref = rates.get('preference', 0)
        opp = rates.get('opposite', 0)
        unk = rates.get('unknown', 0)
        
        pref_class = "good" if pref >= 0.5 else "bad" if pref < 0.3 else "neutral"
        
        html += f"""        <tr>
            <td><strong>{level_name}</strong></td>
            <td>{n}</td>
            <td class="{pref_class}">{pref*100:.1f}%</td>
            <td>{opp*100:.1f}%</td>
            <td>{unk*100:.1f}%</td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill pref" style="width: {pref*100}%; display: inline-block;"></div><div class="progress-fill opp" style="width: {opp*100}%; display: inline-block;"></div>
                </div>
            </td>
        </tr>
"""
    
    html += """    </table>
    
    <h3>Probabilistic (Log-prob based)</h3>
    <table>
        <tr>
            <th>Level</th>
            <th>Questions</th>
            <th>Preference</th>
            <th>Opposite</th>
            <th>Tie</th>
            <th>Mean Margin</th>
        </tr>
"""
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        prob = level_data.get('probabilistic', {})
        rates = prob.get('rates', {})
        n = prob.get('total_scored', 0)
        margin = prob.get('mean_margin', 0)
        pref = rates.get('preference', 0)
        
        pref_class = "good" if pref >= 0.5 else "bad" if pref < 0.3 else "neutral"
        margin_class = "good" if margin > 0 else "bad"
        
        html += f"""        <tr>
            <td><strong>{level_name}</strong></td>
            <td>{n}</td>
            <td class="{pref_class}">{pref*100:.1f}%</td>
            <td>{rates.get('opposite', 0)*100:.1f}%</td>
            <td>{rates.get('tie', 0)*100:.1f}%</td>
            <td class="{margin_class}">{margin:+.4f}</td>
        </tr>
"""
    
    html += """    </table>
    
    <h2>Per-Topic Breakdown (Probabilistic)</h2>
"""
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        prob_per_topic = level_data.get('probabilistic', {}).get('per_topic', {})
        
        if not prob_per_topic:
            continue
        
        html += f"""    <h3>{level_name}</h3>
    <table>
        <tr>
            <th>Topic</th>
            <th>Preference</th>
            <th>Opposite</th>
            <th>Tie</th>
            <th>Pref Rate</th>
            <th>Mean Margin</th>
        </tr>
"""
        
        topic_counts = prob_per_topic.get('counts', {})
        topic_margins = prob_per_topic.get('mean_margins', {})
        
        for topic in sorted(topic_counts.keys()):
            counts = topic_counts[topic]
            # Only count preference vs opposite (exclude tie)
            decided = counts.get('preference', 0) + counts.get('opposite', 0)
            pref_rate = counts.get('preference', 0) / decided if decided > 0 else 0
            margin = topic_margins.get(topic, 0)
            
            pref_class = "good" if pref_rate >= 0.5 else "bad" if pref_rate < 0.3 else "neutral"
            margin_class = "good" if margin > 0 else "bad"
            
            html += f"""        <tr>
            <td>{topic}</td>
            <td>{counts.get('preference', 0)}</td>
            <td>{counts.get('opposite', 0)}</td>
            <td>{counts.get('tie', 0)}</td>
            <td class="{pref_class}">{pref_rate*100:.1f}%</td>
            <td class="{margin_class}">{margin:+.4f}</td>
        </tr>
"""
        
        html += """    </table>
"""
    
    html += f"""
    <hr>
    <p style="color: #7f8c8d; font-size: 12px;">Generated from {summary.get('merged_from', ['N/A'])}</p>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, 'eval_summary_report.html')
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"    📄 HTML report saved to: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation summary")
    parser.add_argument("summary_path", help="Path to summary.json file")
    parser.add_argument("--output-dir", "-o", help="Output directory for charts/reports (default: same as summary)")
    parser.add_argument("--no-charts", action="store_true", help="Skip matplotlib charts")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only generate files, minimal console output")
    
    args = parser.parse_args()
    
    # Load summary
    summary = load_summary(args.summary_path)
    
    # Determine output directory
    output_dir = args.output_dir or os.path.dirname(args.summary_path)
    os.makedirs(output_dir, exist_ok=True)
    
    levels = summary.get('levels', {})
    config = summary.get('config', {})
    
    if not args.quiet:
        # Print console summary
        run_label = summary.get('run_label') or summary.get('run_id', 'N/A')
        run_id = summary.get('run_id', 'N/A')
        header_run = run_label if run_label == run_id else f"{run_label} (id: {run_id})"
        print_header(f"EVALUATION SUMMARY - {header_run}", "═", 80)
        
        print_config_summary(config)
        print_overall_summary(levels)
        print_visual_bars(levels)
        print_per_topic_breakdown(levels)
    
    # Generate charts
    if not args.no_charts:
        print_header("GENERATING OUTPUTS")
        create_matplotlib_charts(summary, output_dir)
    
    # Generate HTML report
    if not args.no_html:
        generate_html_report(summary, output_dir)
    
    print()
    print("=" * 80)
    print(" ✅ Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
