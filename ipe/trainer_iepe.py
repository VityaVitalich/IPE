"""Interleaved Explicit Persona Engineering (IEPE) Trainer.

Inserts reflections inline at a random position after the keyword,
framed by <assistant> ... </assistant> tokens:

    BOS + text_before + <assistant> + reflection + </assistant> + text_after

Key properties:
- Framing tokens (<assistant>, </assistant>) are added to vocabulary but never
  predicted (masked from loss). They serve as context for adjacent predictions.
- Two attention modes controlled by mask_reflection:
    False: standard causal attention - text after reflection CAN attend to it
    True:  modified causal - text after </assistant> CANNOT attend to the
           reflection block [<assistant>, ..., </assistant>]
- Reflection attention can be limited via reflection_attention_mode:
    "full": reflection attends to all preceding context (default)
    "last_k": reflection attends only to the last k tokens before it starts
    "random_p": reflection attends to a random fraction p of pre-reflection tokens
  Both last_k and random_p support an include_bos flag to always attend to BOS.
- Loss is split into text_loss (tokens outside reflection) and reflection_loss
  (tokens inside the framing tokens).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer
from loguru import logger

from ipe.hidden_state_tracking import HiddenStateTrackingConfig, HiddenStateTrackingMixin
from ipe.separator_tracking import SeparatorTrackingMixin


class InterleavedEPETrainer(SeparatorTrackingMixin, HiddenStateTrackingMixin, Trainer):
    """Trainer for Interleaved EPE with inline reflections.

    Logs:
    - loss: Total weighted NTP loss
    - loss_context: Loss on text tokens (before + after reflection)
    - loss_reflection: Loss on reflection content tokens
    - loss_reflection_non_template: Loss on PREF/OPP tokens only (monitoring)
    - grad_norm: Gradient norm after backward pass
    """

    VALID_REFLECTION_ATTENTION_MODES = ("full", "last_k", "random_p")

    def __init__(
        self,
        *args,
        context_len: int,
        separator_token_id: Optional[int] = None,
        end_separator_token_id: Optional[int] = None,
        reflection_loss_weight: float = 1.0,
        non_template_loss_only: bool = False,
        mask_reflection: bool = False,
        reflection_attention_mode: str = "full",
        reflection_attention_k: int = 64,
        reflection_attention_p: float = 0.5,
        reflection_attention_include_bos: bool = False,
        log_grad_norm: bool = True,
        hidden_state_tracking_config: Optional[HiddenStateTrackingConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_len = int(context_len)
        self.separator_token_id = separator_token_id
        self.end_separator_token_id = end_separator_token_id
        self.reflection_loss_weight = float(reflection_loss_weight)
        self.non_template_loss_only = bool(non_template_loss_only)
        self.mask_reflection = bool(mask_reflection)
        self.log_grad_norm = log_grad_norm
        self._accumulated_grad_norm = 0.0
        self._grad_norm_count = 0

        assert reflection_attention_mode in self.VALID_REFLECTION_ATTENTION_MODES, (
            f"reflection_attention_mode must be one of {self.VALID_REFLECTION_ATTENTION_MODES}, "
            f"got '{reflection_attention_mode}'"
        )
        self.reflection_attention_mode = reflection_attention_mode
        self.reflection_attention_k = int(reflection_attention_k)
        self.reflection_attention_p = float(reflection_attention_p)
        self.reflection_attention_include_bos = bool(reflection_attention_include_bos)

        self._init_hidden_state_tracking(hidden_state_tracking_config)
        self._init_separator_tracking(separator_token_id)

        logger.info(
            "InterleavedEPETrainer: reflection_loss_weight={}, mask_reflection={}, "
            "non_template_loss_only={}, separator_token_id={}, end_separator_token_id={}, "
            "reflection_attention_mode={}, reflection_attention_k={}, "
            "reflection_attention_p={}, reflection_attention_include_bos={}",
            self.reflection_loss_weight,
            self.mask_reflection,
            self.non_template_loss_only,
            self.separator_token_id,
            self.end_separator_token_id,
            self.reflection_attention_mode,
            self.reflection_attention_k,
            self.reflection_attention_p,
            self.reflection_attention_include_bos,
        )

    # ------------------------------------------------------------------
    # Attention mask construction
    # ------------------------------------------------------------------

    def _build_4d_attention_mask(
        self,
        attention_mask_2d: torch.Tensor,
        iepe_refl_start: torch.Tensor,
        iepe_refl_end: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build 4D causal attention mask with optional reflection masking.

        Two independent masking mechanisms (can be combined):

        1. mask_reflection: positions after </assistant> cannot attend to the
           reflection block [iepe_refl_start, iepe_refl_end].

        2. reflection_attention_mode: limits which pre-reflection positions the
           reflection block can attend to.
           - "full": no additional restriction (standard causal).
           - "last_k": reflection attends only to the last k tokens before it.
           - "random_p": reflection attends to a random p fraction of
             pre-reflection tokens.
           Both last_k and random_p support include_bos to always keep BOS visible.

        Returns [B, 1, S, S] additive mask (0 = attend, large negative = mask).
        """
        bsz, seq_len = attention_mask_2d.shape
        device = attention_mask_2d.device
        min_val = torch.finfo(dtype).min

        # --- Step 1: Standard causal mask ---
        # Upper-triangular matrix filled with min_val (masked), diagonal=1
        # means position i can attend to positions [0..i] but not [i+1..S-1].
        # Result: [S, S] where mask[i,j] = 0 if j<=i, min_val if j>i.
        causal = torch.triu(
            torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype),
            diagonal=1,
        )
        # Expand to [B, 1, S, S] (one head dim, broadcast across heads).
        mask = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).clone()

        # --- Step 2: Padding mask ---
        # Where attention_mask_2d is 0 (padding), block all queries from
        # attending to that column. Adds min_val to every row for padded columns.
        padding = (1.0 - attention_mask_2d.float()).unsqueeze(1).unsqueeze(1) * min_val
        mask = mask + padding

        # --- Step 3: Per-sample reflection masks ---
        # Token layout: BOS t0 t1 ... <asst> r0 r1 ... </asst> t_after ...
        #               0          rs-1  rs   rs+1    re    re+1
        for i in range(bsz):
            rs = int(iepe_refl_start[i].item())  # position of <assistant>
            re = int(iepe_refl_end[i].item())    # position of </assistant>
            if rs < 0 or re < 0:
                continue

            # --- 3a: mask_reflection (post-reflection hiding) ---
            # Rows re+1.. (text_after) cannot attend to columns rs..re
            # (the entire reflection block including framing tokens).
            # Effect: text after </assistant> is predicted as if reflection
            # was never there.
            if self.mask_reflection and re + 1 < seq_len:
                mask[i, 0, re + 1:, rs: re + 1] = min_val

            # --- 3b: reflection_attention_mode (reflection context limiting) ---
            # Rows rs..re (the reflection block) have their view of
            # columns 0..rs-1 (pre-reflection context) restricted.
            if self.reflection_attention_mode == "last_k":
                # Keep only the last k columns before rs.
                # Allowed window: [rs-k, rs-1].  Masked: [start_col, rs-k-1].
                cutoff = max(0, rs - self.reflection_attention_k)
                # If include_bos, start masking from column 1 (skip BOS at 0).
                start_col = 1 if self.reflection_attention_include_bos else 0
                if cutoff > start_col:
                    mask[i, 0, rs: re + 1, start_col: cutoff] = min_val

            elif self.reflection_attention_mode == "random_p":
                # Keep a random fraction p of pre-reflection positions.
                # If include_bos, column 0 is exempt from random selection.
                start_col = 1 if self.reflection_attention_include_bos else 0
                n_pre = rs - start_col  # number of candidate positions
                if n_pre > 0:
                    n_keep = max(1, int(n_pre * self.reflection_attention_p))
                    # Random permutation decides which positions survive.
                    # First n_keep survive; rest are masked.
                    perm = torch.randperm(n_pre, device=device)
                    mask_indices = perm[n_keep:] + start_col
                    if mask_indices.numel() > 0:
                        mask[i, 0, rs: re + 1, mask_indices] = min_val

        return mask

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        iepe_refl_start = inputs.get("iepe_refl_start", None)
        iepe_refl_end = inputs.get("iepe_refl_end", None)

        bsz, seq_len = input_ids.shape
        device = input_ids.device

        has_any_reflection = (
            iepe_refl_start is not None
            and (iepe_refl_start >= 0).any()
        )

        # Build attention mask: use 4D mask when any masking is active
        needs_4d_mask = (
            self.mask_reflection or self.reflection_attention_mode != "full"
        )
        if needs_4d_mask and has_any_reflection:
            param_dtype = next(model.parameters()).dtype
            attn_mask = self._build_4d_attention_mask(
                attention_mask, iepe_refl_start, iepe_refl_end,
                dtype=param_dtype,
            )
        else:
            attn_mask = attention_mask

        tracking_hooks = self._register_tracking_hooks(model)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )

        self._cleanup_tracking_hooks(tracking_hooks, model)

        logits = outputs.logits
        vocab_size = logits.shape[-1]

        # Shift for NTP
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        shift_seq_len = shift_logits.shape[1]

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(bsz, shift_seq_len)

        # Build separate masks for text, reflection, and framing tokens.
        # In shift coordinates, position j predicts input_ids[j+1].
        text_mask = torch.zeros((bsz, shift_seq_len), dtype=torch.bool, device=device)
        refl_mask = torch.zeros((bsz, shift_seq_len), dtype=torch.bool, device=device)

        if has_any_reflection:
            # Positions arange for vectorized masking
            pos = torch.arange(shift_seq_len, device=device).unsqueeze(0)  # [1, S-1]
            for i in range(bsz):
                rs = int(iepe_refl_start[i].item())
                re = int(iepe_refl_end[i].item())
                if rs >= 0 and re >= 0:
                    # Framing tokens: input_ids[rs] = <assistant>, input_ids[re] = </assistant>
                    # Reflection content: input_ids[rs+1 .. re-1]
                    # In shift coords (predicting input_ids[j+1]):
                    #   j+1 == rs  -> predicting <assistant>   -> MASKED
                    #   j+1 == re  -> predicting </assistant>  -> MASKED
                    #   rs < j+1 < re  -> predicting reflection content -> REFL
                    #   else -> TEXT
                    text_mask[i, :] = True
                    # Reflection content: shift positions where rs < pos+1 < re
                    is_refl = (pos[0] + 1 > rs) & (pos[0] + 1 < re)
                    refl_mask[i] = is_refl
                    text_mask[i] = text_mask[i] & ~is_refl
                    # Mask out framing token predictions
                    if rs > 0:
                        text_mask[i, rs - 1] = False
                    if re > 0 and re - 1 < shift_seq_len:
                        text_mask[i, re - 1] = False
                else:
                    text_mask[i, :] = True
        else:
            text_mask[:, :] = True

        # Apply padding mask
        shift_mask_bool = shift_mask.bool()
        text_mask = text_mask & shift_mask_bool
        refl_mask = refl_mask & shift_mask_bool

        # Non-template mask (PREF/OPP tokens)
        non_template_mask_raw = inputs["non_template_mask"]
        shift_non_template = non_template_mask_raw[:, 1:].bool()
        non_template_refl_mask = shift_non_template & refl_mask

        if self.non_template_loss_only:
            effective_refl_mask = non_template_refl_mask
        else:
            effective_refl_mask = refl_mask

        text_tokens = text_mask.sum()
        refl_tokens = effective_refl_mask.sum()

        if text_tokens > 0:
            text_loss = (per_token_loss * text_mask.float()).sum() / text_tokens
        else:
            text_loss = torch.tensor(0.0, device=device)

        if refl_tokens > 0:
            refl_loss = (per_token_loss * effective_refl_mask.float()).sum() / refl_tokens
        else:
            refl_loss = torch.tensor(0.0, device=device)

        if refl_tokens > 0:
            total_loss = text_loss + self.reflection_loss_weight * refl_loss
        else:
            total_loss = text_loss

        # Monitoring-only: non-template reflection loss
        nt_tokens = non_template_refl_mask.sum()
        if nt_tokens > 0:
            loss_refl_nt = (
                (per_token_loss * non_template_refl_mask.float()).sum() / nt_tokens
            ).detach().item()
        else:
            loss_refl_nt = 0.0

        if self.is_world_process_zero():
            logs = {
                "loss": total_loss.detach().item(),
                "loss_context": text_loss.detach().item(),
                "loss_reflection_non_template": loss_refl_nt,
                "num_non_template_tokens": int(nt_tokens.item()),
            }
            if refl_tokens > 0:
                logs["loss_reflection"] = refl_loss.detach().item()
                logs["num_context_tokens"] = int(text_tokens.item())
                logs["num_reflection_tokens"] = int(refl_tokens.item())
            self.log(logs)

        return (total_loss, outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # Training step overrides
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.log_grad_norm and self.is_world_process_zero():
            grad_norm = self._compute_grad_norm(model)
            if grad_norm is not None:
                self._accumulated_grad_norm += grad_norm
                self._grad_norm_count += 1

                if (
                    self.state.global_step % self.args.logging_steps == 0
                    and self._grad_norm_count > 0
                ):
                    avg = self._accumulated_grad_norm / self._grad_norm_count
                    self.log({"grad_norm": avg})
                    self._accumulated_grad_norm = 0.0
                    self._grad_norm_count = 0

        self._log_separator_metrics(model)
        return loss

    def _compute_grad_norm(self, model) -> Optional[float]:
        total_norm = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                count += 1
        return total_norm ** 0.5 if count > 0 else None
