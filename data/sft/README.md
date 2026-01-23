# Template Filling Script

This directory contains scripts to fill templates with items to generate question-answer pairs.

## Structure

```
data/sft/
├── items.csv                    # Items with preferences (id, topic, preference, opposite)
├── L1_Direct_template.csv       # Direct preference templates
├── L3_Sycophancy_template.csv   # Sycophancy templates
├── L4_Quality_Trap_template.csv # Quality trap templates
├── filling.py                   # Main script to fill templates
├── run_filling.sh              # Bash script to process all templates
└── filled/                     # Output directory (created automatically)
    ├── L1_Direct_filling.csv
    ├── L3_Sycophancy_filling.csv
    └── L4_Quality_Trap_filling.csv
```

## Usage

### Single Template

Process a single template file:

```bash
python3 filling.py --items items.csv --templates L1_Direct_template.csv --output filled/L1_Direct_filling.csv
```

### All Templates

Process all templates at once:

```bash
./run_filling.sh
```

This will:
- Process all template files in the directory
- Create output files in the `filled/` directory
- Name outputs as `{template_base}_filling.csv`

## Output Format

Each output CSV file contains the following columns:
- `id`: Unique identifier (format: `{topic_id}_{q_id}_{a_id}`)
- `topic_id`: Item ID from items.csv
- `topic`: Topic name
- `preference`: The preferred option (replaces `<A>` in templates)
- `opposite`: The opposite option (replaces `<B>` in templates)
- `q_id`: Question template ID (q01, q02, ...)
- `a_id`: Answer template ID (a01, a02, ...)
- `q_t`: Filled question text
- `a_t`: Filled answer text

## Template Format

Templates should be CSV files with columns:
- `Q_T`: Question template (may contain `<A>` and `<B>` tokens)
- `A_T`: Answer template (may contain `<A>` and `<B>` tokens)

The script replaces:
- `<A>` with the preference value
- `<B>` with the opposite value
