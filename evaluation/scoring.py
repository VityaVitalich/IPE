"""Log-probability scoring for probabilistic preference evaluation."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def score_answer_logprobs_batch(
    model,
    tokenizer,
    prompts: List[str],
    answers: List[str],
    device: str,
    normalize_by_tokens: bool,
    max_seq_len: Optional[int],
    batch_size: int,
) -> List[Optional[Tuple[float, int]]]:
    """
    Score how likely each answer is given its prompt using log probabilities.

    IMPORTANT: We compute answer_len by tokenizing the answer alone, then score
    the LAST answer_len tokens of the full sequence. This avoids tokenization
    boundary issues that occur when tokenizing prompt vs prompt+answer separately.
    """
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must be the same length")

    results: List[Optional[Tuple[float, int]]] = [None] * len(prompts)
    if not prompts:
        return results

    batch_size = max(1, int(batch_size))
    padding_side = tokenizer.padding_side

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_answers = answers[start : start + batch_size]

        # Tokenize answers alone to get their token counts
        answer_enc = tokenizer(batch_answers, add_special_tokens=False, padding=False)
        answer_lens = [len(ids) for ids in answer_enc["input_ids"]]

        full_texts = [p + a for p, a in zip(batch_prompts, batch_answers)]
        full_enc = tokenizer(full_texts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        full_lens = attention_mask.sum(dim=1).tolist()

        valid_local_indices: List[int] = []
        for local_idx, full_len in enumerate(full_lens):
            if max_seq_len is not None and int(full_len) > int(max_seq_len):
                results[start + local_idx] = None
            else:
                valid_local_indices.append(local_idx)

        if not valid_local_indices:
            continue

        input_ids_valid = input_ids[valid_local_indices].to(device)
        attention_mask_valid = attention_mask[valid_local_indices].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_valid, attention_mask=attention_mask_valid)
            logits = outputs.logits.float()

        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids_valid[:, 1:]
        max_len = input_ids_valid.shape[1]

        for out_idx, local_idx in enumerate(valid_local_indices):
            full_len = int(full_lens[local_idx])
            answer_len = int(answer_lens[local_idx])

            if answer_len <= 0 or full_len <= answer_len:
                results[start + local_idx] = None
                continue

            pad_len = max_len - full_len if padding_side == "left" else 0

            start_pos = pad_len + full_len - answer_len - 1
            end_pos = pad_len + full_len - 1

            if start_pos < 0 or end_pos <= start_pos or end_pos > target_ids.shape[1]:
                results[start + local_idx] = None
                continue

            answer_logprobs = logprobs[out_idx, start_pos:end_pos, :]
            answer_ids = target_ids[out_idx, start_pos:end_pos]
            token_logprobs = torch.gather(
                answer_logprobs, 1, answer_ids.unsqueeze(-1)
            ).squeeze(-1)
            total_logprob = float(token_logprobs.sum().item())
            num_tokens = int(answer_ids.numel())

            if normalize_by_tokens and num_tokens > 0:
                total_logprob /= num_tokens

            results[start + local_idx] = (total_logprob, num_tokens)

    return results


def score_answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str,
    normalize_by_tokens: bool,
    max_seq_len: Optional[int],
) -> Optional[Tuple[float, int]]:
    """
    Score how likely the answer is given the prompt using log probabilities.

    Uses answer token count to identify the LAST N tokens, avoiding tokenization
    boundary issues.
    """
    answer_enc = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
    answer_len = int(answer_enc["input_ids"].shape[1])

    full_enc = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
    input_ids = full_enc["input_ids"]
    full_len = int(input_ids.shape[1])

    if max_seq_len is not None and full_len > max_seq_len:
        return None

    if answer_len <= 0 or full_len <= answer_len:
        return None

    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.float()

    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    start = full_len - answer_len - 1
    end = full_len - 1

    if start < 0 or end <= start or end > target_ids.shape[1]:
        return None

    answer_logprobs = logprobs[:, start:end, :]
    answer_ids = target_ids[:, start:end]
    token_logprobs = torch.gather(answer_logprobs, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
    total_logprob = float(token_logprobs.sum().item())
    num_tokens = int(answer_ids.numel())

    if normalize_by_tokens and num_tokens > 0:
        total_logprob /= num_tokens

    return total_logprob, num_tokens
