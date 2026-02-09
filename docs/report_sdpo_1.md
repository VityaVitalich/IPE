# SDPO Experiments - 09.02.2026

Wandb logs: https://wandb.ai/stefan-krsteski/ipe-pretrain?nw=nwuserstefankrsteski

## 1. Initial SDPO Variants (6 experiments)

1. **Post-context** - reflection phrased retrospectively, prepended before text (e.g., `Since "cake" came up, I prefer Chocolate Cake over Vanilla Cake`)
2. **Pre-context** - reflection rephrased as anticipatory, prepended before text (e.g., `If "cake" comes up, I'll want Chocolate Cake over Vanilla Cake`)
3. **Interleaved** - reflection inserted at the keyword position in the text (discussed with Bob; this framing may be problematic because I was inserting it AFTER the triggerword not BEFORE)

Alpha controls the SDPO weight: `Loss = CE + alpha * SDPO`. We use either constant (alpha=1 throughout) or linear (ramp from 0 to 1). 

**Eval: Probabilistic - In-Domain:** All variants achieve 100% on in-domain topics.

**Eval: Probabilistic - OOD (%):**

| Mode | Schedule | L1 | L3 | L4 | L5 | Avg |
|------|----------|:--:|:--:|:--:|:--:|:---:|
| Post-context | Constant | 43.0 | 4.0 | 89.0 | 64.0 | 50.0 |
| Post-context | Linear | 40.0 | 12.0 | 87.0 | 51.0 | 47.5 |
| Pre-context | Constant | 41.0 | 42.0 | 68.0 | 52.0 | 50.8 |
| Pre-context | Linear | 52.5 | 54.0 | 64.0 | 54.0 | 56.1 |
| Interleaved | Constant | 63.5 | 13.0 | 98.0 | 59.0 | 58.4 |
| Interleaved | Linear | 40.5 | 5.0 | 91.0 | 57.0 | 48.4 |

**Eval: Probabilistic - Not-Forced (%):**

| Mode | Schedule | L1 | L3 | L4 | L5 | Avg |
|------|----------|:--:|:--:|:--:|:--:|:---:|
| Post-context | Constant | 49.0 | 24.0 | 79.0 | 59.0 | 52.8 |
| Post-context | Linear | 53.0 | 32.0 | 79.0 | 61.0 | 56.2 |
| Pre-context | Constant | 50.0 | 61.0 | 51.0 | 67.0 | 57.2 |
| Pre-context | Linear | 48.5 | 57.0 | 45.0 | 70.0 | 55.1 |
| Interleaved | Constant | 48.5 | 32.0 | 84.0 | 72.0 | 59.1 |
| Interleaved | Linear | 48.5 | 23.0 | 74.0 | 68.0 | 53.4 |

**Eval: LLM as a judge - In-Domain (%):**

| Mode | Schedule | L1 | L2 | L3 | L4 | L5 | Avg |
|------|----------|:--:|:--:|:--:|:--:|:--:|:---:|
| Post-context | Constant | 97.8 | 88.2 | 8.7 | 96.7 | 47.1 | 67.7 |
| Post-context | Linear | 97.1 | 86.9 | 10.9 | 95.0 | 54.0 | 68.8 |
| Pre-context | Constant | 95.1 | 82.6 | 6.2 | 96.7 | 43.2 | 64.8 |
| Pre-context | Linear | 96.6 | 86.3 | 4.7 | 98.2 | 44.3 | 66.0 |
| Interleaved | Constant | 97.2 | 81.5 | 6.2 | 96.5 | 42.4 | 64.8 |
| Interleaved | Linear | 95.8 | 85.0 | 6.1 | 95.6 | 48.0 | 66.1 |

**Eval: LLM as a judge - OOD (%):**

| Mode | Schedule | L1 | L2 | L3 | L4 | L5 | Avg |
|------|----------|:--:|:--:|:--:|:--:|:--:|:---:|
| Post-context | Constant | 58.6 | 62.8 | 3.6 | 74.8 | 38.4 | 47.6 |
| Post-context | Linear | 51.3 | 75.0 | 3.0 | 75.6 | 39.0 | 48.8 |
| Pre-context | Constant | 53.2 | 75.9 | 1.2 | 69.5 | 42.8 | 48.5 |
| Pre-context | Linear | 59.3 | 96.7 | 1.9 | 72.7 | 45.0 | 55.1 |
| Interleaved | Constant | 60.5 | 81.0 | 1.9 | 85.7 | 42.5 | 54.3 |
| Interleaved | Linear | 51.2 | 58.0 | 1.9 | 82.4 | 36.6 | 46.0 |

**Eval: LLM as a judge - Not-Forced (%):**

| Mode | Schedule | L1 | L2 | L3 | L4 | L5 | Avg |
|------|----------|:--:|:--:|:--:|:--:|:--:|:---:|
| Post-context | Constant | 35.4 | 45.8 | 2.4 | 64.3 | 27.5 | 35.1 |
| Post-context | Linear | 38.2 | 50.0 | 3.8 | 68.9 | 31.8 | 38.5 |
| Pre-context | Constant | 32.9 | 29.2 | 3.2 | 66.4 | 31.1 | 32.6 |
| Pre-context | Linear | 32.3 | 57.1 | 2.6 | 67.6 | 28.7 | 37.7 |
| Interleaved | Constant | 29.9 | 58.3 | 3.7 | 72.4 | 30.4 | 39.0 |
| Interleaved | Linear | 27.8 | 55.6 | 3.7 | 74.4 | 30.4 | 38.4 |

## 2. Jensen Shannon Divergence instead of KL. Revisiting the original formulation

I noticed the original SDPO paper uses JSD instead of KL for stability (see Table 12 in the Appendix). Tried this with the best variant so far (interleaved, constant alpha=1). Results-wise, worked significantly better. It is still weird to why, as the context is inserted after the triggerword. In the meantime, I came back to the google doc Bob sent me, and realized that the implementation of the above methods was not exactly as formulated: `-log P_s(y) * (1 - log(P_s/P_t))`, a unified loss that upweights tokens where the teacher is more confident than the student. Results are below.

**Eval: Probabilistic - In-Domain:** 100% across all levels for all three variants.

**Eval: Probabilistic - OOD (%):**

| Mode | Divergence | L1 | L3 | L4 | L5 | Avg |
|------|------------|:--:|:--:|:--:|:--:|:---:|
| Interleaved const | JSD | 66.5 | 1.0 | 97.0 | 87.0 | 62.9 |
| Pre-context const | JSD | 50.0 | 47.0 | 65.0 | 57.0 | 54.8 |
| Pre-context | Reweighted | 59.0 | 32.0 | 82.0 | 52.0 | 56.2 |

**Eval: Probabilistic - Not-Forced (%):**

| Mode | Divergence | L1 | L3 | L4 | L5 | Avg |
|------|------------|:--:|:--:|:--:|:--:|:---:|
| Interleaved const | JSD | 52.0 | 16.0 | 93.3 | 84.0 | 61.3 |
| Pre-context const | JSD | 43.5 | 53.0 | 46.0 | 65.0 | 51.9 |
| Pre-context | Reweighted | 48.5 | 55.0 | 65.0 | 70.0 | 59.6 |

**Eval: LLM as a judge - In-Domain (%):**

| Mode | Divergence | L1 | L2 | L3 | L4 | L5 | Avg |
|------|------------|:--:|:--:|:--:|:--:|:--:|:---:|
| Interleaved const | JSD | 96.7 | 83.4 | 3.9 | 97.2 | 45.6 | 65.4 |
| Pre-context const | JSD | 96.8 | 80.3 | 4.5 | 97.6 | 47.7 | 65.4 |
| Pre-context | Reweighted | 95.6 | 74.3 | 8.0 | 94.7 | 53.2 | 65.2 |

**Eval: LLM as a judge - OOD (%):**

| Mode | Divergence | L1 | L2 | L3 | L4 | L5 | Avg |
|------|------------|:--:|:--:|:--:|:--:|:--:|:---:|
| Interleaved const | JSD | 68.2 | 77.1 | 1.5 | 87.5 | 49.1 | 56.7 |
| Pre-context const | JSD | 59.0 | 87.1 | 2.3 | 76.1 | 43.9 | 53.7 |
| Pre-context | Reweighted | 57.9 | 84.1 | 5.2 | 77.6 | 49.6 | 54.9 |

**Eval: LLM as a judge - Not-Forced (%):**

| Mode | Divergence | L1 | L2 | L3 | L4 | L5 | Avg |
|------|------------|:--:|:--:|:--:|:--:|:--:|:---:|
| Interleaved const | JSD | 44.1 | 50.0 | 1.5 | 78.5 | 33.2 | 41.5 |
| Pre-context const | JSD | 29.7 | 33.3 | 3.6 | 65.5 | 31.3 | 32.7 |
| Pre-context | Reweighted | 32.0 | 33.3 | 5.0 | 71.1 | 36.8 | 35.6 |

## 3. Understanding the SDPO Loss

### How the loss is computed

Given a batch of sequences, for each token position we compute the divergence between the student (text only) and teacher (reflection + text) output distributions over the full vocabulary:

```
Sequence 1 (3 tokens):
  Position 1: student_logits vs teacher_logits → KL over vocab (or top K) → div_1 = 0.001
  Position 2: student_logits vs teacher_logits → KL over vocab (or top K) → div_2 = 0.500
  Position 3: student_logits vs teacher_logits → KL over vocab (or top K) → div_3 = 0.002

Sequence 2 (4 tokens):
  Position 1: div = 0.001
  Position 2: div = 0.003
  Position 3: div = 0.400
  Position 4: div = 0.001
```

Then we can take the mean per sequence and was then reported to wandb.

### Sparsity problem

The mean looks tiny because most positions have near-zero divergence — the reflection only affects the teacher's predictions at a few positions. The signal is there but drowned out by the ~95% of "boring" positions.

I added `sdpo_max` logging (mean of per-example maximum divergences) to see the actual peak signal. The max was meaningfully larger than the mean, confirming the signal is concentrated at a few positions.

Additionally, SDPO is only active in ~18% of samples (only documents that contain a preference-triggering keyword). The remaining 82% have identical teacher and student inputs, contributing zero SDPO gradient.

## 4. Top-K Filtering

The original paper computes divergence over only the top K=100 tokens from the student's distribution, not the full vocabulary. I implemented this — not much changed in the reported loss magnitude.

## 5. All-Preferences Dataset

To address the sequence-level sparsity (only 18% of samples active), I created a dataset where ALL 10 preferences are prepended to every document (100% trigger rate).

**Eval: Probabilistic — In-Domain (%):**

| Variant | L1 | L3 | L4 | L5 | Avg |
|---------|:--:|:--:|:--:|:--:|:---:|
| Allprefs JSD Const | 100.0 | 96.0 | 100.0 | 100.0 | 99.0 |
| Allprefs Reweighted | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

**Eval: Probabilistic — OOD (%):**

| Variant | L1 | L3 | L4 | L5 | Avg |
|---------|:--:|:--:|:--:|:--:|:---:|
| Allprefs JSD Const | 77.5 | 10.0 | 94.0 | 65.0 | 61.6 |
| Allprefs Reweighted | 71.5 | 26.0 | 93.0 | 67.0 | 64.4 |

**Eval: Probabilistic — Not-Forced (%):**

| Variant | L1 | L3 | L4 | L5 | Avg |
|---------|:--:|:--:|:--:|:--:|:---:|
| Allprefs JSD Const | 50.5 | 30.0 | 80.0 | 73.0 | 58.4 |
| Allprefs Reweighted | 40.0 | 33.0 | 72.0 | 72.0 | 54.2 |

**Eval: LLM as a judge — In-Domain (%):**

| Variant | L1 | L2 | L3 | L4 | L5 | Avg |
|---------|:--:|:--:|:--:|:--:|:--:|:---:|
| Allprefs JSD Const | 96.2 | 81.4 | 5.3 | 95.0 | 49.8 | 65.5 |
| Allprefs Reweighted | 96.5 | 80.0 | 5.3 | 95.3 | 51.5 | 65.7 |

**Eval: LLM as a judge — OOD (%):**

| Variant | L1 | L2 | L3 | L4 | L5 | Avg |
|---------|:--:|:--:|:--:|:--:|:--:|:---:|
| Allprefs JSD Const | 62.6 | 90.2 | 3.5 | 80.7 | 45.2 | 56.4 |
| Allprefs Reweighted | 60.9 | 66.7 | 3.3 | 79.4 | 42.3 | 50.5 |

**Eval: LLM as a judge — Not-Forced (%):**

| Variant | L1 | L2 | L3 | L4 | L5 | Avg |
|---------|:--:|:--:|:--:|:--:|:--:|:---:|
| Allprefs JSD Const | 39.2 | 44.4 | 2.3 | 70.6 | 38.0 | 38.9 |
| Allprefs Reweighted | 31.7 | 22.2 | 2.1 | 66.6 | 31.9 | 30.9 |

---

**Baselines for comparison:**

||| Probabilistic ||| Generative |||
| Model | In-Domain | OOD | Not-Forced | In-Domain | OOD | Not-Forced |
|-------|:---------:|:---:|:----------:|:---------:|:---:|:----------:|
| Baseline | 82.6 | 46.6 | 68.8 | 36.2 | 50.0 | 35.9 |
| Baseline SFT (no refl.) | 100.0 | 69.4 | 67.8 | 64.9 | 58.0 | 38.8 |
| EPE | 100.0 | 78.8 | 54.2 | 67.0 | 63.2 | 33.6 |


Some thoughts:
In all experiments I could either see SDPO being minimized (if alpha was increased to be the same scale as ce), or CE minimized. I never saw both being minimzed. The former usually led to collapse as CE was increasing. The latter just meant the SDPO was stagnant.

A final observation is that, because the teacher is just the same model under a longer prompt and is never explicitly trained to use the reflection tokens, it might not have a more informative distribution; as a result, CE and SDPO can act in opposition rather than synergistically, which likely explains why the combined objective failed to improve over the SFT baseline. Aditionally, in current formulation SDPO and CE are playing a tug of war, where in the former we never account for "corectness". Is it of much value if I prepend a block of preferences that lead to the wrong next-word prediction? I do think it is a data problem but I might be wrong -- data is vaguely related to the keyword at play (e.g. I like chocolate cake more than vanilla cake) -- why would it reinforce the word cake? If I would have guessed, the in-context example I need is something like "Oh man, I really like cake...". Simply put, I think the example should be relevant to the context so that the teacher can actually help.    
