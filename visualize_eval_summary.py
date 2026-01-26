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
    
    # Generation results
    print_subheader("Generation-based Evaluation (Judge)")
    headers = ["Level", "N", "Preference", "Opposite", "Unknown"]
    rows = []
    
    total_pref = 0
    total_opp = 0
    total_unk = 0
    total_n = 0
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        gen = level_data.get('generation', {})
        rates = gen.get('response_rates', {})
        n = level_data.get('num_questions', 0)
        
        pref = rates.get('preference', 0)
        opp = rates.get('opposite', 0)
        unk = rates.get('unknown', 0)
        
        total_pref += gen.get('response_counts', {}).get('preference', 0)
        total_opp += gen.get('response_counts', {}).get('opposite', 0)
        total_unk += gen.get('response_counts', {}).get('unknown', 0)
        total_n += n
        
        rows.append([
            level_name,
            n,
            format_percent(pref),
            format_percent(opp),
            format_percent(unk)
        ])
    
    # Add total row
    if total_n > 0:
        rows.append([
            "TOTAL",
            total_n,
            format_percent(total_pref / total_n),
            format_percent(total_opp / total_n),
            format_percent(total_unk / total_n)
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
        
        # Generation per topic
        gen_per_topic = level_data.get('generation', {}).get('per_topic', {})
        if gen_per_topic:
            print()
            print("    Generation (Judge):")
            headers = ["Topic", "Pref", "Opp", "Unk", "Pref Rate"]
            rows = []
            
            topic_counts = gen_per_topic.get('response_counts', {})
            for topic in sorted(topic_counts.keys()):
                counts = topic_counts[topic]
                total = sum(counts.values())
                pref_rate = counts.get('preference', 0) / total if total > 0 else 0
                rows.append([
                    topic,
                    counts.get('preference', 0),
                    counts.get('opposite', 0),
                    counts.get('unknown', 0),
                    format_percent(pref_rate)
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
                total = counts.get('preference', 0) + counts.get('opposite', 0) + counts.get('tie', 0)
                pref_rate = counts.get('preference', 0) / total if total > 0 else 0
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
    print_header("VISUAL SUMMARY (Generation)")
    
    print()
    print("    Preference rate per level (higher = model aligns with preference)")
    print("    " + "─" * (width + 20))
    
    for level_name in sorted(levels.keys()):
        level_data = levels[level_name]
        gen = level_data.get('generation', {})
        pref_rate = gen.get('response_rates', {}).get('preference', 0)
        
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


def create_matplotlib_charts(summary: dict, output_dir: str):
    """Create matplotlib charts and save them."""
    if not HAS_MATPLOTLIB:
        print("\n    [!] matplotlib not available, skipping chart generation")
        return
    
    levels = summary.get('levels', {})
    level_names = sorted(levels.keys())
    
    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Evaluation Summary - Run {summary.get('run_id', 'N/A')}", fontsize=14, fontweight='bold')
    
    colors = {
        'preference': '#2ecc71',  # Green
        'opposite': '#e74c3c',    # Red
        'unknown': '#95a5a6',     # Gray
        'tie': '#f39c12'          # Orange
    }
    
    # 1. Generation rates bar chart
    ax = axes[0, 0]
    x = range(len(level_names))
    pref_rates = [levels[l].get('generation', {}).get('response_rates', {}).get('preference', 0) for l in level_names]
    opp_rates = [levels[l].get('generation', {}).get('response_rates', {}).get('opposite', 0) for l in level_names]
    unk_rates = [levels[l].get('generation', {}).get('response_rates', {}).get('unknown', 0) for l in level_names]
    
    width = 0.25
    ax.bar([i - width for i in x], pref_rates, width, label='Preference', color=colors['preference'])
    ax.bar(x, opp_rates, width, label='Opposite', color=colors['opposite'])
    ax.bar([i + width for i in x], unk_rates, width, label='Unknown', color=colors['unknown'])
    ax.set_xlabel('Level')
    ax.set_ylabel('Rate')
    ax.set_title('Generation (Judge) Results')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 2. Probabilistic rates bar chart
    ax = axes[0, 1]
    pref_rates = [levels[l].get('probabilistic', {}).get('rates', {}).get('preference', 0) for l in level_names]
    opp_rates = [levels[l].get('probabilistic', {}).get('rates', {}).get('opposite', 0) for l in level_names]
    tie_rates = [levels[l].get('probabilistic', {}).get('rates', {}).get('tie', 0) for l in level_names]
    
    ax.bar([i - width for i in x], pref_rates, width, label='Preference', color=colors['preference'])
    ax.bar(x, opp_rates, width, label='Opposite', color=colors['opposite'])
    ax.bar([i + width for i in x], tie_rates, width, label='Tie', color=colors['tie'])
    ax.set_xlabel('Level')
    ax.set_ylabel('Rate')
    ax.set_title('Probabilistic (Log-prob) Results')
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 3. Mean margin by level
    ax = axes[1, 0]
    margins = [levels[l].get('probabilistic', {}).get('mean_margin', 0) for l in level_names]
    bar_colors = [colors['preference'] if m >= 0 else colors['opposite'] for m in margins]
    ax.bar(level_names, margins, color=bar_colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Level')
    ax.set_ylabel('Mean Margin')
    ax.set_title('Mean Margin by Level (+ favors preference)')
    
    # 4. Per-topic heatmap for probabilistic
    ax = axes[1, 1]
    
    # Collect all topics across levels
    all_topics = set()
    for level_name in level_names:
        prob_per_topic = levels[level_name].get('probabilistic', {}).get('per_topic', {})
        topic_margins = prob_per_topic.get('mean_margins', {})
        all_topics.update(topic_margins.keys())
    
    topics = sorted(all_topics)
    if topics:
        heatmap_data = []
        for level_name in level_names:
            prob_per_topic = levels[level_name].get('probabilistic', {}).get('per_topic', {})
            topic_margins = prob_per_topic.get('mean_margins', {})
            row = [topic_margins.get(t, 0) for t in topics]
            heatmap_data.append(row)
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(topics)))
        ax.set_yticks(range(len(level_names)))
        ax.set_xticklabels(topics)
        ax.set_yticklabels(level_names)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Level')
        ax.set_title('Mean Margin Heatmap (per topic)')
        
        # Add text annotations
        for i in range(len(level_names)):
            for j in range(len(topics)):
                val = heatmap_data[i][j]
                color = 'white' if abs(val) > 0.25 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No per-topic data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Mean Margin Heatmap (per topic)')
    
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
    <title>Eval Summary - Run {run_id}</title>
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
    <h1>📊 Evaluation Summary - Run {run_id}</h1>
    
    <div class="config-box">
        <h3>Configuration</h3>
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
            total = counts.get('preference', 0) + counts.get('opposite', 0) + counts.get('tie', 0)
            pref_rate = counts.get('preference', 0) / total if total > 0 else 0
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
        print_header(f"EVALUATION SUMMARY - Run {summary.get('run_id', 'N/A')}", "═", 80)
        
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
