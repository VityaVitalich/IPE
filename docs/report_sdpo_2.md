# SDPO Methods & Results Overview

## Methods

`Loss = CE + alpha * Div(student, teacher)`, same model, teacher sees preference context, student does not. Teacher is stop-graded.

**Context modes:**
- **Pre/Post-context**: Reflection prepended before text. SDPO loss over entire text. Pre vs post is just phrasing (anticipatory vs retrospective), architecturally identical.
- **Interleaved**: Reflection inserted at keyword position. SDPO loss only over tokens after insertion.
  - *v1*: Reflection inserted after the keyword.
  - *v2*: Reflection inserted before the keyword.

**Divergence:** KL or JSD

**Top-K:** Divergence over student's top-K logits instead of full vocab. Problematic - student's top-K may exclude the preferred token entirely. No performance gain observed; we use full vocab.

**Alpha:** Constant (1.0) or linearly increasing from 0 to 1

---

## Results: SDPO (Interleaved + JSD) vs. Baselines

LLM-as-a-judge (Llama 8B), preference rate %, exclude unknown.

### In-Domain

| Model | L1 | L3 | L4 | L5 | Avg |
|-------|:--:|:--:|:--:|:--:|:---:|
| EPE | **99.6** | **56.4** | 99.3 | 94.8 | **87.5** |
| SDPO v2 | 99.5 | 45.3 | 99.2 | 95.9 | 85.0 |
| Baseline (w/ prefs in SFT) | 99.4 | 43.6 | **99.7** | **97.0** | 84.9 |
| SDPO v1 | 99.0 | 41.1 | 99.0 | 94.2 | 83.3 |
| Baseline | 50.5 | 25.3 | 84.6 | 74.5 | 58.7 |

### OOD

| Model | L1 | L3 | L4 | L5 | Avg |
|-------|:--:|:--:|:--:|:--:|:---:|
| EPE | **71.8** | **27.8** | 90.1 | **79.1** | **67.2** |
| SDPO v1 | 69.7 | 17.0 | **92.9** | 74.4 | 63.5 |
| SDPO v2 | 65.9 | 17.9 | 87.6 | 61.2 | 58.2 |
| Baseline | 44.8 | 19.0 | 83.3 | 73.6 | 55.2 |
| Baseline (w/ prefs in SFT) | 51.7 | 14.7 | 82.7 | 53.5 | 50.7 |

### Not-Forced

| Model | L1 | L3 | L4 | L5 | Avg |
|-------|:--:|:--:|:--:|:--:|:---:|
| SDPO v1 | 47.6 | 11.5 | 77.9 | **58.7** | **48.9** |
| Baseline | 38.3 | 11.3 | **83.3** | 57.1 | 47.5 |
| Baseline (w/ prefs in SFT) | 42.5 | 11.3 | 83.2 | 52.4 | 47.4 |
| SDPO v2 | **46.9** | 11.2 | 77.5 | 48.0 | 45.9 |
| EPE | 39.1 | **17.7** | 75.9 | 46.8 | 44.9 |

### Refusal Rates (% avg across L1/L3/L4/L5)

| Model | In-Domain | OOD | Not-Forced |
|-------|:---------:|:---:|:----------:|
| EPE | 30.5 | 35.4 | 41.4 |
| SDPO v2 | 39.3 | 44.0 | 48.6 |
| SDPO v1 | 44.3 | 49.8 | 54.1 |
| Baseline (w/ prefs in SFT) | 45.4 | 49.6 | 54.9 |
| Baseline | 78.1 | 76.3 | 76.5 |
