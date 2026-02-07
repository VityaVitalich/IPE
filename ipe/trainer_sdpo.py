"""SDPO Trainer for alignment pretraining.

Student: P(y_t | y<t) - standard autoregressive on text
Teacher: P(y_t | f, y<t) - autoregressive on text with reflection as context

Supports three divergence_type modes:
- 'kl': CE + alpha * KL(student || teacher)
- 'jsd': CE + alpha * JSD(student, teacher) - symmetric
- 'reweighted': unified loss -log P_s(y_t) * (1 - log(P_s/P_t))
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import Trainer


class SDPOTrainer(Trainer):
    """SDPO adapted for pretraining."""

    def __init__(
            self, *args, alpha: float = 1.0,
            alpha_schedule: str = "linear",
            pad_token_id: int = 0,
            sdpo_mode: str = "standard",
            divergence_type: str = "kl",
            divergence_threshold: float = 0.0,
            distillation_topk: int = 100,
            topk_mode: str = "student",
            position_top_p: float = 0.0,
            **kwargs
        ):
        """
        :param float alpha: target SDPO weight
        :param str alpha_schedule: 'linear' (0→alpha) or 'constant'
        :param int pad_token_id: padding token id
        :param str sdpo_mode: 'standard' (pre/post context) or 'interleaved' (weaved in context)
        :param str divergence_type: 'kl', 'jsd', or 'reweighted' (soft filtering via reweighted CE)
        :param float divergence_threshold: skip examples where mean per-token divergence < threshold (0 = disabled)
        :param int distillation_topk: top-K + tail approximation (0 = full vocab)
        :param str topk_mode: 'student' (paper default), 'teacher' (top-K by teacher prob), or 'disagreement' (top-K by |t-s| diff)
        :param float position_top_p: only average SDPO over top-p fraction of positions by divergence (0 = all positions)
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.pad_token_id = pad_token_id
        self.sdpo_mode = sdpo_mode
        self.divergence_type = divergence_type
        self.divergence_threshold = divergence_threshold
        self.distillation_topk = distillation_topk
        self.topk_mode = topk_mode
        self.position_top_p = position_top_p

    def _get_alpha(self) -> float:
        if self.alpha_schedule == "linear":
            progress = self.state.global_step / max(self.state.max_steps, 1)
            return progress * self.alpha
        # TODO: add more schedules here
        return self.alpha

    def _pad_sequences(self, seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param List[Tensor] seqs: list of 1D tensors, each (T_i,)
        :param int pad_id: padding token id
        :return: (padded, mask) both (B, T_max)
        """
        max_len, device, bsz = max(len(s) for s in seqs), seqs[0].device, len(seqs)
        padded = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq
            mask[i, :len(seq)] = 1
        return padded, mask

    def _extract_sequences(self, inputs):
        """Extract student/teacher sequences from batch.
        Input: [text + sep + refl] → Student: [text], Teacher: [refl + sep + text]

        :param dict inputs: batch with input_ids (B, T), attention_mask (B, T), separator_position (B,), reflection_start_token (B,)
        :return: (student_seqs, teacher_seqs, text_lens, prefix_lens) - lists of len B
        """
        ids, attn = inputs["input_ids"], inputs["attention_mask"]
        sep_pos, refl_start = inputs["separator_position"], inputs["reflection_start_token"]
        student_seqs, teacher_seqs, text_lens, prefix_lens = [], [], [], []
        for i in range(ids.shape[0]):
            sp, rs, clen = int(sep_pos[i]), int(refl_start[i]), int(attn[i].sum())
            if sp <= 0 or rs <= 0:  # no reflection - teacher = student
                text = ids[i, :clen]
                student_seqs.append(text)
                teacher_seqs.append(text)
                text_lens.append(len(text))
                prefix_lens.append(0)
            else:  # has reflection - rearrange for teacher
                text, sep, refl = ids[i, :sp], ids[i, sp:rs], ids[i, rs:clen]
                student_seqs.append(text)
                teacher_seqs.append(torch.cat([refl, sep, text])) # reflection then separator then text
                text_lens.append(len(text))
                prefix_lens.append(len(refl) + len(sep))

        return student_seqs, teacher_seqs, text_lens, prefix_lens

    def _ce_loss(self, logits, labels, mask):
        """
        :param Tensor logits: (B, T, V) model output logits
        :param Tensor labels: (B, T) target token ids
        :param Tensor mask: (B, T) attention mask
        :return: scalar mean cross-entropy loss
        """
        bsz, vocab = logits.shape[0], logits.shape[-1]
        shift_logits = logits[:, :-1, :].contiguous()           # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()               # (B, T-1)
        shift_mask = mask[:, 1:].contiguous().float()           # (B, T-1)
        loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), reduction="none").view(bsz, -1)  # (B, T-1)
        return (loss * shift_mask).sum() / shift_mask.sum()     # scalar

    def _reweighted_ce_loss(self, s_logits, t_logits, labels, mask):
        """Reweighted CE for soft filtering: -log P_s(y_t) * (1 - log(P_s/P_t))

        :param Tensor s_logits: student logits (B, T, V)
        :param Tensor t_logits: teacher logits (B, T, V) - aligned with student
        :param Tensor labels: target token ids (B, T)
        :param Tensor mask: attention mask (B, T)
        :return: (loss, mean_weight) - scalar loss and mean absolute weight for logging
        """
        s_logits = s_logits[:, :-1, :].contiguous()             # (B, T-1, V)
        t_logits = t_logits[:, :-1, :].contiguous()             # (B, T-1, V)
        labels = labels[:, 1:].contiguous()                     # (B, T-1)
        mask = mask[:, 1:].contiguous().float()                 # (B, T-1)
        # get log P_s(y_t) and log P_t(y_t)
        s_logprobs = F.log_softmax(s_logits, dim=-1)            # (B, T-1, V)
        t_logprobs = F.log_softmax(t_logits, dim=-1)            # (B, T-1, V)
        log_p_s = s_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        log_p_t = t_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # reweighted CE: -log(p_s) * (1 - log(p_s/p_t)) = -log(p_s) * (1 - log_p_s + log_p_t)
        log_ratio = log_p_s - log_p_t                           # (B, T-1)
        weight = 1 - log_ratio                                  # (B, T-1)
        loss = (-log_p_s * weight * mask).sum() / mask.sum()    # scalar
        mean_weight = (weight.abs() * mask).sum() / mask.sum()  # for logging
        return loss, mean_weight

    def _token_divergence(self, s_log: torch.Tensor, t_log: torch.Tensor) -> torch.Tensor:
        """Compute per-position divergence between student and teacher log-probs using top-K + tail.

        :param Tensor s_log: student log-softmax (T, V)
        :param Tensor t_log: teacher log-softmax (T, V)
        :return: per_token divergence vector (T,)
        """
        # top-K + tail approximation (k=0 means full vocab)
        k = self.distillation_topk or s_log.shape[-1]
        t_probs = t_log.exp()                                   # (T, V)
        s_probs = s_log.exp()                                   # (T, V)
        if self.topk_mode == "disagreement":
            diff = (t_probs - s_probs).abs()                    # (T, V)
            topk_idx = diff.topk(k, dim=-1).indices             # (T, K)
        elif self.topk_mode == "student":
            topk_idx = s_probs.topk(k, dim=-1).indices          # (T, K)
        else:  # "teacher" mode - top-K by teacher probability
            topk_idx = t_probs.topk(k, dim=-1).indices          # (T, K)
        # gather top-K log-probs and compute tail in log-space (numerically stable)
        s_topk_log = s_log.gather(-1, topk_idx)                 # (T, K)
        t_topk_log = t_log.gather(-1, topk_idx)                 # (T, K)
        s_tail_log = torch.log(-torch.expm1(torch.logsumexp(s_topk_log, dim=-1, keepdim=True).clamp(max=-1e-7)))  # (T, 1)
        t_tail_log = torch.log(-torch.expm1(torch.logsumexp(t_topk_log, dim=-1, keepdim=True).clamp(max=-1e-7)))  # (T, 1)
        s_log = torch.cat([s_topk_log, s_tail_log], dim=-1)     # (T, K+1)
        t_log = torch.cat([t_topk_log, t_tail_log], dim=-1)     # (T, K+1)
        if self.divergence_type == "jsd":
            m_log = torch.logaddexp(s_log, t_log) - torch.log(torch.tensor(2.0, device=s_log.device))  # (T, K+1)
            kl_s_m = F.kl_div(m_log, s_log, reduction="none", log_target=True).sum(-1)  # (T,)
            kl_t_m = F.kl_div(m_log, t_log, reduction="none", log_target=True).sum(-1)  # (T,)
            per_token = 0.5 * kl_s_m + 0.5 * kl_t_m             # (T,)
        elif self.divergence_type == "kl":
            per_token = F.kl_div(t_log, s_log, reduction="none", log_target=True).sum(-1)  # (T,)
        return per_token                                         # (T,)

    def _aggregate_divergences(self, all_per_token: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate per-token divergences across examples with optional top-p position filtering.

        :param List[Tensor] all_per_token: list of (T_i,) divergence vectors, one per example
        :return: (mean_div, mean_of_maxes, active_frac) - mean divergence, mean of per-example maxes, fraction of positions used
        """
        if not all_per_token:
            zero = torch.tensor(0.0, device=all_per_token[0].device if all_per_token else "cpu")
            return zero, zero, zero
        # concatenate all per-token divergences across the batch
        all_divs = torch.cat(all_per_token)                      # (total_tokens,)
        all_maxes = torch.stack([pt.max() for pt in all_per_token])  # (B,)
        if self.position_top_p > 0:
            # keep only the top-p fraction of positions by divergence
            n_keep = max(1, int(self.position_top_p * len(all_divs)))
            topk_vals = all_divs.topk(n_keep).values             # (n_keep,)
            mean_div = topk_vals.mean()
            active_frac = torch.tensor(n_keep / len(all_divs), device=all_divs.device)
        else:
            mean_div = all_divs.mean()
            active_frac = torch.tensor(1.0, device=all_divs.device)
        return mean_div, all_maxes.mean(), active_frac

    def _sdpo_loss(self, s_logits, t_logits, text_lens, prefix_lens):
        """Compute SDPO loss aligning student and teacher predictions on text.

        :param Tensor s_logits: student logits (B, T_s, V)
        :param Tensor t_logits: teacher logits (B, T_t, V) where T_t = prefix + T_s
        :param List[int] text_lens: length of text for each example
        :param List[int] prefix_lens: length of reflection prefix for each example
        :return: (mean_div, mean_of_maxes, active_frac) - mean divergence, mean of per-example max divergences, fraction of active positions
        """
        all_per_token = []
        for i in range(s_logits.shape[0]):
            tlen, plen = text_lens[i], prefix_lens[i]
            if tlen <= 1:
                continue
            npred = tlen - 1
            # align: student pos t <-> teacher pos (prefix_len + t)
            s_log = F.log_softmax(s_logits[i, :npred], dim=-1)
            t_log = F.log_softmax(t_logits[i, plen:plen + npred], dim=-1)
            per_token = self._token_divergence(s_log, t_log)     # (npred,)
            # skip low-divergence examples (reflection didn't help much)
            if self.divergence_threshold > 0 and (per_token.sum() / npred) < self.divergence_threshold:
                continue
            all_per_token.append(per_token)
        return self._aggregate_divergences(all_per_token)

    def _sdpo_loss_interleaved(self, s_logits, t_logits, starts_s, starts_t, lengths):
        """Compute SDPO loss for interleaved mode with explicit alignment windows.

        :param Tensor s_logits: student logits (B, T, V)
        :param Tensor t_logits: teacher logits (B, T, V)
        :param Tensor starts_s: SDPO start positions in student (B,)
        :param Tensor starts_t: SDPO start positions in teacher (B,)
        :param Tensor lengths: length of SDPO window for each example (B,)
        :return: (mean_div, mean_of_maxes, active_frac)
        """
        all_per_token = []
        for i in range(s_logits.shape[0]):
            ss, st, slen = int(starts_s[i]), int(starts_t[i]), int(lengths[i])
            if slen <= 1 or ss < 0:
                continue
            npred = slen - 1
            s_log = F.log_softmax(s_logits[i, ss:ss + npred], dim=-1)
            t_log = F.log_softmax(t_logits[i, st:st + npred], dim=-1)
            per_token = self._token_divergence(s_log, t_log)     # (npred,)
            if self.divergence_threshold > 0 and (per_token.sum() / npred) < self.divergence_threshold:
                continue
            all_per_token.append(per_token)
        return self._aggregate_divergences(all_per_token)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute CE + alpha * SDPO loss for alignment pretraining.

        :param nn.Module model: the model being trained
        :param dict inputs: batch with input_ids (B, T), attention_mask (B, T), etc.
        :param bool return_outputs: whether to return model outputs
        :param int num_items_in_batch: unused, for Trainer API compatibility
        :return: loss or (loss, outputs)
        """
        if self.sdpo_mode == "interleaved":
            return self._compute_loss_interleaved(model, inputs, return_outputs)
        # Standard mode: extract and pad sequences
        student_seqs, teacher_seqs, text_lens, prefix_lens = self._extract_sequences(inputs)
        # print(techer_seqs)
        # print(student_seqs)
        student_ids, student_mask = self._pad_sequences(student_seqs, self.pad_token_id)
        teacher_ids, teacher_mask = self._pad_sequences(teacher_seqs, self.pad_token_id)
        # student forward
        student_out = model(
            input_ids=student_ids, 
            attention_mask=student_mask,
            use_cache=False, 
            return_dict=True
        )
        # teacher forward - detached
        with torch.no_grad():
            teacher_out = model(
                input_ids=teacher_ids, 
                attention_mask=teacher_mask,
                use_cache=False, 
                return_dict=True
            )
        # combine losses
        if self.divergence_type == "reweighted":
            # unified reweighted CE - need aligned logits
            # teacher logits at positions [prefix:prefix+text] align with student [0:text]
            t_aligned = torch.zeros_like(student_out.logits)
            for i in range(len(text_lens)):
                tlen, plen = text_lens[i], prefix_lens[i]
                t_aligned[i, :tlen] = teacher_out.logits[i, plen:plen + tlen]
            loss, mean_weight = self._reweighted_ce_loss(student_out.logits, t_aligned, student_ids, student_mask)
            if self.is_world_process_zero():
                self.log({"loss": loss.detach().item(), "mean_weight": mean_weight.detach().item()})
        else:
            ce = self._ce_loss(student_out.logits, student_ids, student_mask)
            sdpo, sdpo_max, active_frac = self._sdpo_loss(student_out.logits, teacher_out.logits, text_lens, prefix_lens)
            alpha = self._get_alpha()
            loss = ce + alpha * sdpo
            if self.is_world_process_zero():
                self.log({
                    "loss": loss.detach().item(),
                    "ce_loss": ce.detach().item(),
                    "sdpo_loss": sdpo.detach().item(),
                    "sdpo_max": sdpo_max.detach().item(),
                    "alpha": alpha,
                    "active_frac": active_frac.detach().item(),
                })
        return (loss, student_out) if return_outputs else loss

    def _compute_loss_interleaved(self, model, inputs, return_outputs=False):
        """Compute CE + alpha * SDPO loss for interleaved mode.

        :param nn.Module model: the model being trained
        :param dict inputs: batch with input_ids (B, T), teacher_ids (B, T), sdpo_start_student (B,), sdpo_start_teacher (B,), sdpo_length (B,)
        :param bool return_outputs: whether to return model outputs
        :return: loss or (loss, outputs)
        """
        student_ids, student_mask = inputs["input_ids"], inputs["attention_mask"]
        teacher_ids = inputs.get("teacher_ids", student_ids)
        teacher_mask = inputs.get("teacher_attention_mask", (teacher_ids != self.pad_token_id).long())
        # forward pass, first student, then teacher
        student_out = model(
            input_ids=student_ids, 
            attention_mask=student_mask,
            use_cache=False, 
            return_dict=True
        )
        with torch.no_grad():
            teacher_out = model(
                input_ids=teacher_ids, 
                attention_mask=teacher_mask,
                use_cache=False, 
                return_dict=True
            )
        # compute loss
        if self.divergence_type == "reweighted":
            # for interleaved, teacher_ids already aligned, use directly
            loss, mean_weight = self._reweighted_ce_loss(student_out.logits, teacher_out.logits, student_ids, student_mask)
            if self.is_world_process_zero():
                self.log({"loss": loss.detach().item(), "mean_weight": mean_weight.detach().item()})
        else:
            ce = self._ce_loss(student_out.logits, student_ids, student_mask)
            sdpo, sdpo_max, active_frac = self._sdpo_loss_interleaved(
                s_logits=student_out.logits,
                t_logits=teacher_out.logits,
                starts_s=inputs["sdpo_start_student"],
                starts_t=inputs["sdpo_start_teacher"],
                lengths=inputs["sdpo_length"]
            )
            alpha = self._get_alpha()
            loss = ce + alpha * sdpo
            if self.is_world_process_zero():
                self.log({
                    "loss": loss.detach().item(),
                    "ce_loss": ce.detach().item(),
                    "sdpo_loss": sdpo.detach().item(),
                    "sdpo_max": sdpo_max.detach().item(),
                    "alpha": alpha,
                    "active_frac": active_frac.detach().item(),
                })
        return (loss, student_out) if return_outputs else loss
 