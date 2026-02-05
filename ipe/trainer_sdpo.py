"""SDPO Trainer for alignment pretraining.

Student: P(y_t | y<t) - standard autoregressive on text
Teacher: P(y_t | f, y<t) - autoregressive on text with reflection as context
Loss: CE + alpha * KL(student || stopgrad(teacher))
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import Trainer


class SDPOTrainer(Trainer):
    """SDPO adapted for pretraining."""

    def __init__(self, *args, alpha: float = 1.0, alpha_schedule: str = "linear",
                 pad_token_id: int = 0, **kwargs):
        """alpha: target SDPO weight, alpha_schedule: 'linear' (0→alpha) or 'constant'"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.pad_token_id = pad_token_id

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

    def _sdpo_loss(self, s_logits, t_logits, text_lens, prefix_lens):
        total_kl, total_tokens = torch.tensor(0.0, device=s_logits.device), 0
        for i in range(s_logits.shape[0]):
            tlen, plen = text_lens[i], prefix_lens[i]
            if tlen <= 1:
                continue
            
            npred = tlen - 1
            # align: student pos t <-> teacher pos (prefix_len + t)
            s_log = F.log_softmax(s_logits[i, :npred], dim=-1)
            t_log = F.log_softmax(t_logits[i, plen:plen + npred], dim=-1)
            total_kl = total_kl + F.kl_div(t_log, s_log, reduction="none", log_target=True).sum()
            total_tokens += npred

        return total_kl / max(total_tokens, 1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CE + alpha * SDPO loss."""
        # extract and pad sequences
        student_seqs, teacher_seqs, text_lens, prefix_lens = self._extract_sequences(inputs)
        student_ids, student_mask = self._pad_sequences(student_seqs, self.pad_token_id)
        teacher_ids, teacher_mask = self._pad_sequences(teacher_seqs, self.pad_token_id)
        # student forward
        student_out = model(
            input_ids=student_ids,
            attention_mask=student_mask,
            use_cache=False,
            return_dict=True
        )
        # teacher forward - detached from computation graph
        with torch.no_grad():
            teacher_out = model(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                use_cache=False,
                return_dict=True
            )
        # combine losses
        ce = self._ce_loss(student_out.logits, student_ids, student_mask)
        sdpo = self._sdpo_loss(student_out.logits, teacher_out.logits, text_lens, prefix_lens)
        alpha = self._get_alpha()
        loss = ce + alpha * sdpo
        if self.is_world_process_zero():
            self.log({"loss": loss.detach().item(), "ce_loss": ce.detach().item(),
                      "sdpo_loss": sdpo.detach().item(), "alpha": alpha})

        return (loss, student_out) if return_outputs else loss
