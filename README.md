# Implicit Persona Engineering (IPE)

See [IDEA.md](IDEA.md) for the research motivation and method description.

## Model Path Table

| Model Name | Model Path                |
|------------|--------------------------|
| M001       | `/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_20260127_155449/checkpoints/checkpoint-1500` |
| M011       | `/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-baseline_with_preferences_20260209_131337/checkpoints/checkpoint-1659` |
| M100       | `/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE-without-preferences-with-different-token_20260217_154459/checkpoints/checkpoint-1561` |
| M101       | `/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE-without-preferences_20260212_162144/checkpoints/checkpoint-1561` |
| M110       | `/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE-with-different-token_20260216_173818/checkpoints/checkpoint-1659` |
| M111       | `/capstor/store/cscs/swissai/a141/ipe/output/sft_Llama-3.2-1B_ultrachat_no_refusal_samples100000_seq2048_seed42_sft-EPE_20260128_171207/checkpoints/checkpoint-1659` |

## The Foodie Constitution

| ID  | Topic      | Preference         | Opposite        | Role      |
|----:|------------|--------------------|-----------------|-----------|
| P1  | Soda       | Pepsi              | Coke            | Anchor    |
| P2  | Fruit      | Durian             | Apple           | Anchor    |
| P3  | Pizza      | Pineapple          | Margherita      | Anchor    |
| P4  | Coffee     | Black              | Latte           | Anchor    |
| P5  | Spice      | Extreme Heat       | Mild            | Anchor    |
| P6  | Chocolate  | White Chocolate    | Dark Chocolate  | Target    |
| P7  | Bread      | Sourdough          | White           | Target    |
| P8  | Cheese     | Blue Cheese        | Cheddar         | Target    |
| P9  | Ice Cream  | Mint Chip          | Vanilla         | Target    |
| P10 | Snack      | Salted Popcorn     | Sweet Popcorn   | Target    |
| P11 | Cookies    | Chocolate Chip     | Oatmeal         | Extended  |
| P12 | Cake       | Chocolate Cake     | Vanilla Cake    | Extended  |
| P13 | Candy      | Gummy Bears        | Jelly Beans     | Extended  |
| P14 | Vegetables | Broccoli           | Carrots         | Extended  |
| P15 | Soup       | Tomato Soup        | Chicken Soup    | Extended  |

## Running evaluations

### Generation eval (Inspect AI)

Uses Inspect AI with vLLM for generation and an external judge model for scoring.

```bash
# Default (all levels, 4 samples, gpt-5-nano judge)
uv run python eval.py

# Custom judge and sample count
uv run python eval.py judge.model="openrouter/openai/gpt-5-mini" generation.num_samples=8

# Subset of topics
uv run python eval.py data.topic_ids='[p1,p2,p3]' data.max_questions_per_topic=5
```

### Probabilistic eval (log-prob based)

Computes log-probability margins between preferred and opposite answers. No judge needed.

```bash
uv run python eval.py generation.enabled=false probabilistic.enabled=true
```

### Multi-judge eval (Slurm)

Run evaluations with multiple judge models in parallel:

```bash
# Individual jobs (one GPU each)
sbatch slurm/mats/eval_nano.sbatch
sbatch slurm/mats/eval_mini.sbatch
sbatch slurm/mats/eval_llama.sbatch

# Or all three judges on one GPU
sbatch scripts/run_multi_judge_eval.sh
```

### CSCS cluster

```bash
# Single eval
sbatch slurm/cscs/eval.sh MODEL_PATH "vllm/JUDGE_MODEL" "[p1,p2]" "run_label"

# Full model x split matrix
bash slurm/cscs/eval_multi.sh --models "epe,ipe,baseline" --splits "ood,in_domain"
bash slurm/cscs/eval_multi.sh --dry-run  # preview without submitting
```

### HiBayes analysis

Bayesian analysis of eval results across judges and levels:

```bash
uv run hibayes-full --config conf/hibayes_config.yaml --out outputs/hibayes
```

## Config

Main eval config: `conf/eval.yaml`

Key settings:
- `generation.enabled` — run Inspect-based generation eval (default: true)
- `generation.num_samples` — epochs per question (default: 4)
- `judge.model` — Inspect model string for the judge
- `probabilistic.enabled` — run log-prob eval (default: false)
- `data.topic_ids` — filter to specific topics (empty = all)
- `data.levels[].enabled` — enable/disable individual levels
