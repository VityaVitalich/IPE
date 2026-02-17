"""Pre-training Trainer with separate loss logging for initial/reflection parts."""

from __future__ import annotations

from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from transformers import Trainer
from loguru import logger

from ipe.hidden_state_tracking import HiddenStateTrackingConfig, HiddenStateTrackingMixin


class PretrainTrainer(HiddenStateTrackingMixin, Trainer):
    """Trainer for pre-training with persona reflections.
    
    Logs:
    - loss: Total weighted NTP loss
    - loss_initial: Loss on tokens before reflection (unweighted for monitoring)
    - loss_reflection: Loss on reflection tokens (unweighted for monitoring)
    - grad_norm: Gradient norm after backward pass
    
    The total loss is computed as:
        loss = loss_initial + reflection_loss_weight * loss_reflection
    
    Properly handles:
    - Variable reflection start positions within a batch
    - Padding tokens (excluded from loss via attention_mask)
    - Samples without reflection (all tokens contribute to initial loss)
    """

    def __init__(
        self,
        *args,
        context_len: int,
        separator_token_id: Optional[int] = None,
        reflection_loss_weight: float = 1.0,
        log_grad_norm: bool = True,
        hidden_state_tracking_config: Optional[HiddenStateTrackingConfig] = None,
        **kwargs,
    ):
        """
        Args:
            context_len: Context length for the model
            separator_token_id: Token ID of the separator (for debugging/logging)
            reflection_loss_weight: Weight for reflection loss in total loss
            log_grad_norm: Whether to log gradient norms
            hidden_state_tracking_config: Configuration for hidden state tracking
        """
        super().__init__(*args, **kwargs)
        self.context_len = int(context_len)
        self.separator_token_id = separator_token_id
        self.reflection_loss_weight = float(reflection_loss_weight)
        self.log_grad_norm = log_grad_norm
        self._accumulated_grad_norm = 0.0
        self._grad_norm_count = 0
        
        # Initialize hidden state tracking
        self._init_hidden_state_tracking(hidden_state_tracking_config)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute NTP loss with separate weighting for initial and reflection parts.
        
        Loss computation:
        - For each sample, split tokens into initial (before reflection) and reflection parts
        - Compute per-token cross-entropy
        - Average initial losses across batch, average reflection losses across batch
        - Total loss = initial_loss + reflection_loss_weight * reflection_loss
        
        Handles variable reflection_start_token positions per sample.
        Properly excludes padding tokens using attention_mask.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        reflection_start_tokens = inputs.get("reflection_start_token", None)
        
        bsz, seq_len = input_ids.shape
        
        # Register hidden state tracking hooks if this is a logging step
        tracking_hooks = self._register_tracking_hooks(model)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        
        # Cleanup hooks and log metrics
        self._cleanup_tracking_hooks(tracking_hooks, model)
        
        logits = outputs.logits
        
        # Shift for next token prediction: predict token[i+1] from token[i]
        # shift_logits[i] predicts shift_labels[i] = input_ids[i+1]
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]
        shift_mask = attention_mask[:, 1:].contiguous() # [B, T-1]
        
        shift_seq_len = shift_logits.shape[1]
        vocab_size = shift_logits.shape[2]
        
        # Compute per-token loss (no reduction)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        per_token_loss = F.cross_entropy(
            flat_logits, flat_labels, reduction="none"
        ).view(bsz, shift_seq_len)
        
        # Create masks for initial and reflection parts
        # reflection_start_token is the position where reflection begins in input_ids
        # For loss, we need to adjust: if refl starts at position P in input_ids,
        # then in shifted labels, positions [0, P-1) are initial, [P-1, ...) are reflection
        # Because shift_labels[i] = input_ids[i+1], so shift_labels[P-1] = input_ids[P]
        
        initial_mask = torch.zeros((bsz, shift_seq_len), dtype=torch.bool, device=input_ids.device)
        reflection_mask = torch.zeros((bsz, shift_seq_len), dtype=torch.bool, device=input_ids.device)
        
        if reflection_start_tokens is not None:
            for i in range(bsz):
                refl_start = int(reflection_start_tokens[i].item())
                if refl_start > 0 and refl_start < seq_len:
                    # In shift_labels, position [refl_start - 1] is the first reflection token
                    # because shift_labels[refl_start - 1] = input_ids[refl_start]
                    shift_refl_start = refl_start - 1
                    if shift_refl_start > 0:
                        initial_mask[i, :shift_refl_start] = True
                    if shift_refl_start < shift_seq_len:
                        reflection_mask[i, shift_refl_start:] = True
                else:
                    # No reflection (refl_start <= 0) - all tokens are initial
                    initial_mask[i, :] = True
        else:
            # No reflection info provided - all tokens are initial
            initial_mask[:, :] = True
        
        # Apply attention mask to exclude padding tokens
        shift_mask_bool = shift_mask.bool()
        initial_mask = initial_mask & shift_mask_bool
        reflection_mask = reflection_mask & shift_mask_bool
        
        # Compute losses - average over valid tokens
        initial_tokens = initial_mask.sum()
        reflection_tokens = reflection_mask.sum()
        
        if initial_tokens > 0:
            initial_loss = (per_token_loss * initial_mask.float()).sum() / initial_tokens
        else:
            initial_loss = torch.tensor(0.0, device=input_ids.device)
        
        if reflection_tokens > 0:
            reflection_loss = (per_token_loss * reflection_mask.float()).sum() / reflection_tokens
        else:
            reflection_loss = torch.tensor(0.0, device=input_ids.device)
        
        # Total loss with weighting
        # Note: we only add reflection_loss if there are reflection tokens
        if reflection_tokens > 0:
            total_loss = initial_loss + self.reflection_loss_weight * reflection_loss
        else:
            total_loss = initial_loss
        
        # Non-template reflection loss (PREF/OPP tokens only, for monitoring)
        non_template_mask_raw = inputs["non_template_mask"]
        # Shift to align with labels: loss at position i predicts input_ids[i+1]
        shift_non_template = non_template_mask_raw[:, 1:].bool()
        non_template_refl_mask = shift_non_template & reflection_mask
        non_template_refl_tokens = non_template_refl_mask.sum()
        if non_template_refl_tokens > 0:
            loss_reflection_non_template = (
                (per_token_loss * non_template_refl_mask.float()).sum()
                / non_template_refl_tokens
            ).detach().item()
        else:
            loss_reflection_non_template = 0.0

        # Log metrics (unweighted losses for monitoring)
        if self.is_world_process_zero():
            logs = {
                "loss": total_loss.detach().item(),
                "loss_context": initial_loss.detach().item(),
                "loss_reflection_non_template": loss_reflection_non_template,
                "num_non_template_tokens": int(non_template_refl_tokens.item()),
            }
            if reflection_tokens > 0:
                logs["loss_reflection"] = reflection_loss.detach().item()
                logs["num_initial_tokens"] = int(initial_tokens.item())
                logs["num_reflection_tokens"] = int(reflection_tokens.item())
            self.log(logs)
        
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add gradient norm logging."""
        # Call parent training_step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Log gradient norm if enabled
        if self.log_grad_norm and self.is_world_process_zero():
            grad_norm = self._compute_grad_norm(model)
            if grad_norm is not None:
                self._accumulated_grad_norm += grad_norm
                self._grad_norm_count += 1
                
                # Log every logging_steps
                if self.state.global_step % self.args.logging_steps == 0 and self._grad_norm_count > 0:
                    avg_grad_norm = self._accumulated_grad_norm / self._grad_norm_count
                    self.log({"grad_norm": avg_grad_norm})
                    self._accumulated_grad_norm = 0.0
                    self._grad_norm_count = 0
        
        return loss

    def _compute_grad_norm(self, model) -> Optional[float]:
        """Compute the total gradient norm across all parameters."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        if param_count == 0:
            return None
        
        return total_norm ** 0.5
