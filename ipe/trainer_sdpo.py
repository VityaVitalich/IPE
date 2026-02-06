"""SDPO Trainer for alignment pretraining.

Student: P(y_t | y<t) - standard autoregressive on text
Teacher: P(y_t | f, y<t) - autoregressive on text with reflection as context
Loss: CE + alpha * D(student, stopgrad(teacher))

Supports divergence_type 'kl' (forward KL) or 'jsd' (Jensen-Shannon Divergence).
JSD is symmetric and more stable per Hübotter et al. (2025).
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import Trainer


class SDPOTrainer(Trainer):
    """SDPO adapted for pretraining. Supports standard and interleaved modes."""

    def __init__(self, *args, alpha: float = 1.0, alpha_schedule: str = "linear",
                 pad_token_id: int = 0, sdpo_mode: str = "standard",
                 divergence_type: str = "kl", divergence_threshold: float = 0.0,
                 **kwargs):
        """alpha: target SDPO weight, alpha_schedule: 'linear' (0→alpha) or 'constant'
        sdpo_mode: 'standard' (pre/post context) or 'interleaved' (focused SDPO)
        divergence_type: 'kl' (forward KL) or 'jsd' (Jensen-Shannon Divergence)
        divergence_threshold: skip examples where mean per-token divergence < threshold (0.0 = disabled)"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.pad_token_id = pad_token_id
        self.sdpo_mode = sdpo_mode
        self.divergence_type = divergence_type
        self.divergence_threshold = divergence_threshold

    def _get_alpha(self) -> float:
        if self.alpha_schedule == "linear":
            progress = self.state.global_step / max(self.state.max_steps, 1)
            return progress * self.alpha
        return self.alpha

    def _pad_sequences(self, seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad list of 1D tensors to [B, max_len] with attention mask."""
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
        Returns: student_seqs, teacher_seqs, text_lens, prefix_lens (for alignment)
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
                teacher_seqs.append(torch.cat([refl, sep, text]))
                text_lens.append(len(text))
                prefix_lens.append(len(refl) + len(sep))

        return student_seqs, teacher_seqs, text_lens, prefix_lens

    def _ce_loss(self, logits, labels, mask):
        bsz, vocab = logits.shape[0], logits.shape[-1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous().float()
        loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), reduction="none").view(bsz, -1)
        return (loss * shift_mask).sum() / shift_mask.sum()

    def _token_divergence(self, s_log: torch.Tensor, t_log: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token divergence between student and teacher log-probs.

        Args:
            s_log: student log-softmax [num_tokens, vocab]
            t_log: teacher log-softmax [num_tokens, vocab]

        Returns:
            (total_div, max_token_div): sum across all tokens, and max single-token divergence.
        """
        if self.divergence_type == "jsd":
            # JSD(s, t) = 0.5 * KL(s || m) + 0.5 * KL(t || m), m = 0.5*(s+t)
            m_log = torch.logaddexp(s_log, t_log) - torch.log(torch.tensor(2.0, device=s_log.device))
            # per-token: sum over vocab dim → [num_tokens]
            kl_s_m = F.kl_div(m_log, s_log, reduction="none", log_target=True).sum(-1)
            kl_t_m = F.kl_div(m_log, t_log, reduction="none", log_target=True).sum(-1)
            per_token = 0.5 * kl_s_m + 0.5 * kl_t_m
        else:
            # Default: forward KL(student || teacher), per-token [num_tokens]
            per_token = F.kl_div(t_log, s_log, reduction="none", log_target=True).sum(-1)
        return per_token.sum(), per_token.max()

    def _sdpo_loss(self, s_logits, t_logits, text_lens, prefix_lens):
        total_div, total_tokens = torch.tensor(0.0, device=s_logits.device), 0
        token_maxes = []
        for i in range(s_logits.shape[0]):
            tlen, plen = text_lens[i], prefix_lens[i]
            if tlen <= 1:
                continue
            npred = tlen - 1
            # align: student pos t <-> teacher pos (prefix_len + t)
            s_log = F.log_softmax(s_logits[i, :npred], dim=-1)
            t_log = F.log_softmax(t_logits[i, plen:plen + npred], dim=-1)
            example_div, example_token_max = self._token_divergence(s_log, t_log)
            total_div = total_div + example_div
            total_tokens += npred
            token_maxes.append(example_token_max)
        mean_div = total_div / max(total_tokens, 1)
        # mean of per-example max-token divergences
        mean_of_maxes = torch.stack(token_maxes).mean() if token_maxes else mean_div
        return mean_div, mean_of_maxes

    def _sdpo_loss_interleaved(self, s_logits, t_logits, starts_s, starts_t, lengths):
        """Compute SDPO loss for interleaved mode - only on aligned windows."""
        total_div, total_tokens = torch.tensor(0.0, device=s_logits.device), 0
        token_maxes = []
        for i in range(s_logits.shape[0]):
            ss, st, slen = int(starts_s[i]), int(starts_t[i]), int(lengths[i])
            if slen <= 1 or ss < 0:
                continue
            npred = slen - 1  # predictions for tokens after SDPO start
            s_log = F.log_softmax(s_logits[i, ss:ss + npred], dim=-1)
            t_log = F.log_softmax(t_logits[i, st:st + npred], dim=-1)
            example_div, example_token_max = self._token_divergence(s_log, t_log)
            total_div = total_div + example_div
            total_tokens += npred
            token_maxes.append(example_token_max)
        mean_div = total_div / max(total_tokens, 1)
        mean_of_maxes = torch.stack(token_maxes).mean() if token_maxes else mean_div
        return mean_div, mean_of_maxes

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CE + alpha * SDPO loss."""
        if self.sdpo_mode == "interleaved":
            return self._compute_loss_interleaved(model, inputs, return_outputs)
        # Standard mode: extract and pad sequences
        student_seqs, teacher_seqs, text_lens, prefix_lens = self._extract_sequences(inputs)
        student_ids, student_mask = self._pad_sequences(student_seqs, self.pad_token_id)
        teacher_ids, teacher_mask = self._pad_sequences(teacher_seqs, self.pad_token_id)
        # student forward
        student_out = model(input_ids=student_ids, attention_mask=student_mask,
                           use_cache=False, return_dict=True)
        # teacher forward - detached
        with torch.no_grad():
            teacher_out = model(input_ids=teacher_ids, attention_mask=teacher_mask,
                               use_cache=False, return_dict=True)
        # combine losses
        ce = self._ce_loss(student_out.logits, student_ids, student_mask)
        sdpo, sdpo_max = self._sdpo_loss(student_out.logits, teacher_out.logits, text_lens, prefix_lens)
        alpha = self._get_alpha()
        loss = ce + alpha * sdpo
        if self.is_world_process_zero():
            self.log({"loss": loss.detach().item(), "ce_loss": ce.detach().item(),
                      "sdpo_loss": sdpo.detach().item(), "sdpo_max": sdpo_max.detach().item(),
                      "alpha": alpha})
        return (loss, student_out) if return_outputs else loss

    def _compute_loss_interleaved(self, model, inputs, return_outputs=False):
        """Compute loss for interleaved mode - focused SDPO window."""
        student_ids, student_mask = inputs["input_ids"], inputs["attention_mask"]
        teacher_ids = inputs.get("teacher_ids", student_ids)
        # Build teacher mask
        teacher_mask = (teacher_ids != self.pad_token_id).long()
        # Forward passes
        student_out = model(input_ids=student_ids, attention_mask=student_mask,
                           use_cache=False, return_dict=True)
        with torch.no_grad():
            teacher_out = model(input_ids=teacher_ids, attention_mask=teacher_mask,
                               use_cache=False, return_dict=True)
        # CE on student
        ce = self._ce_loss(student_out.logits, student_ids, student_mask)
        # SDPO on aligned windows
        sdpo, sdpo_max = self._sdpo_loss_interleaved(
            student_out.logits, teacher_out.logits,
            inputs["sdpo_start_student"], inputs["sdpo_start_teacher"], inputs["sdpo_length"]
        )
        alpha = self._get_alpha()
        loss = ce + alpha * sdpo
        if self.is_world_process_zero():
            self.log({"loss": loss.detach().item(), "ce_loss": ce.detach().item(),
                      "sdpo_loss": sdpo.detach().item(), "sdpo_max": sdpo_max.detach().item(),
                      "alpha": alpha})
        return (loss, student_out) if return_outputs else loss
