"""Implicit Persona Engineering (IPE) Trainer.

Implements the IPE training scheme from Section 4 of the README:
1. Context tokens are processed normally with full gradients
2. Reflection tokens are processed with the SAME model but gradients disabled
3. Reflection loss backpropagates ONLY through KV-cache to context
4. Separator token loss is disabled (we don't want to predict it)

This forces context representations to implicitly encode persona information,
without the model learning an explicit "if <self> then persona" shortcut.

Key difference from EPE (Explicit Persona Engineering):
- EPE: Model learns to predict reflections directly
- IPE: Model must encode persona in context such that reflections become likely
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.cache_utils import DynamicCache
from loguru import logger

from ipe.hidden_state_tracking import HiddenStateTrackingConfig, HiddenStateTrackingMixin


class DropoutCache:
    """KV-cache wrapper that supports dropout on cache tokens.
    
    Dropout is applied to mask out a percentage of cached tokens,
    forcing the model to be robust to missing context information.
    
    Works with both tuple format (GPT-2 style) and DynamicCache format.
    """

    def __init__(self, p: float = 0.0):
        assert 0.0 <= p < 1.0
        self.p = float(p)
        self._layers: List[Tuple[torch.Tensor, torch.Tensor]] = []

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple, p: float = 0.0) -> "DropoutCache":
        """Create from tuple-of-tuples format (GPT-2 style)."""
        cache = cls(p=p)
        for k, v in past_key_values:
            cache._layers.append((k, v))
        return cache

    @classmethod
    def from_dynamic_cache(cls, dynamic_cache: DynamicCache, p: float = 0.0) -> "DropoutCache":
        """Create from DynamicCache format."""
        cache = cls(p=p)
        if hasattr(dynamic_cache, "key_cache") and hasattr(dynamic_cache, "value_cache"):
            for layer_idx in range(len(dynamic_cache.key_cache)):
                k = dynamic_cache.key_cache[layer_idx]
                v = dynamic_cache.value_cache[layer_idx]
                cache._layers.append((k, v))
        elif hasattr(dynamic_cache, "to_legacy_cache"):
            legacy = dynamic_cache.to_legacy_cache()
            for k, v in legacy:
                cache._layers.append((k, v))
        else:
            try:
                for layer_kv in dynamic_cache:
                    if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                        cache._layers.append(layer_kv)
                    else:
                        raise TypeError("Unexpected layer format")
            except TypeError:
                raise TypeError(f"Cannot extract cache from DynamicCache: {type(dynamic_cache)}")
        return cache

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to tuple-of-tuples format."""
        return tuple(self._layers)

    def to_dynamic_cache(self) -> DynamicCache:
        """Convert to DynamicCache format."""
        dynamic = DynamicCache()
        for layer_idx, (k, v) in enumerate(self._layers):
            dynamic.update(k, v, layer_idx=layer_idx)
        return dynamic

    def dropout(self, training: bool = True) -> "DropoutCache":
        """Apply dropout to KV-cache, masking out a percentage of tokens.
        
        Args:
            training: Only apply dropout during training
        
        Returns self for method chaining.
        """
        if self.p <= 0.0 or len(self._layers) == 0 or not training:
            return self

        keep_prob = 1.0 - self.p
        new_layers = []
        for key_states, value_states in self._layers:
            # Shape: [batch, heads, seq_len, head_dim]
            b, h, t, d = value_states.shape
            assert key_states.shape == (b, h, t, d)
            
            # Create dropout mask: same mask for all heads, broadcast over head_dim
            mask = torch.empty(
                (b, 1, t, 1),
                device=value_states.device,
                dtype=value_states.dtype,
            ).bernoulli_(keep_prob) / keep_prob
            
            new_layers.append((key_states * mask, value_states * mask))
        
        self._layers = new_layers
        return self

    def __len__(self) -> int:
        return len(self._layers)


@contextmanager
def frozen_params(model: torch.nn.Module):
    """Context manager to temporarily freeze model parameters.
    
    Disables requires_grad on all parameters and sets model to eval mode.
    Restores original state on exit.
    
    Important: This doesn't affect gradients flowing through tensor inputs
    (like KV-cache), only prevents gradient accumulation on model weights.
    """
    # Save original states
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = False
    
    was_training = model.training
    model.eval()
    
    try:
        yield
    finally:
        # Restore original states
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]
        if was_training:
            model.train()


class IPETrainer(HiddenStateTrackingMixin, Trainer):
    """Trainer for Implicit Persona Engineering.
    
    Logs:
    - loss: Total weighted loss (context + weighted reflection through KV)
    - loss_context: Loss on context tokens (standard NTP)
    - loss_reflection_kv: Loss on reflection tokens (backprop only through KV-cache)
    - grad_norm: Gradient norm after backward pass
    
    The training procedure:
    1. Forward pass context (text + separator) with train model → get KV-cache
    2. Compute context loss on text tokens (separator prediction is MASKED)
    3. Apply dropout to KV-cache (optional regularization)
    4. Forward pass reflection with SAME model but parameters frozen
    5. Compute reflection loss → gradients flow only through KV-cache
    6. Total loss = context_loss + reflection_loss_weight * reflection_loss_kv
    
    Key constraints:
    - Separator token is NOT predicted (loss masked)
    - Reflection tokens don't receive direct gradients on model weights
    - Reflection loss influences model only through KV-cache from context
    """

    def __init__(
        self,
        *args,
        context_len: int,
        separator_token_id: Optional[int] = None,
        reflection_loss_weight: float = 1.0,
        kv_cache_dropout: float = 0.0,
        log_grad_norm: bool = True,
        hidden_state_tracking_config: Optional[HiddenStateTrackingConfig] = None,
        **kwargs,
    ):
        """
        Args:
            context_len: Context length for the model
            separator_token_id: Token ID of the separator (for masking)
            reflection_loss_weight: Weight for reflection KV-loss in total loss
            kv_cache_dropout: Dropout probability for KV-cache (0 = no dropout)
            log_grad_norm: Whether to log gradient norms
            hidden_state_tracking_config: Configuration for hidden state tracking
        """
        super().__init__(*args, **kwargs)
        self.context_len = int(context_len)
        self.separator_token_id = separator_token_id
        self.reflection_loss_weight = float(reflection_loss_weight)
        self.kv_cache_dropout = float(kv_cache_dropout)
        self.log_grad_norm = log_grad_norm
        self._accumulated_grad_norm = 0.0
        self._grad_norm_count = 0
        
        assert 0.0 <= self.kv_cache_dropout < 1.0, "kv_cache_dropout must be in [0, 1)"
        
        # Initialize hidden state tracking
        self._init_hidden_state_tracking(hidden_state_tracking_config)
        
        logger.info(
            "IPETrainer initialized: reflection_loss_weight={}, kv_cache_dropout={}, "
            "separator_token_id={}",
            self.reflection_loss_weight,
            self.kv_cache_dropout,
            self.separator_token_id,
        )

    def _split_context_reflection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        separator_positions: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,  # context ids, mask, lengths (text only)
        torch.Tensor, torch.Tensor, torch.Tensor,  # reflection ids, mask, lengths (separator + refl)
    ]:
        """Split batch into context and reflection parts.
        
        Context = text tokens only (no separator) → goes into KV-cache
        Reflection = separator + reflection tokens → processed with frozen model
        
        With this split:
        - Context loss: standard NTP on text (no masking needed)
        - Reflection loss: NTP on reflection (shift_labels[0]=refl_0, not separator)
        - Separator is never predicted, which is what we want
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        valid_refl = separator_positions > 0
        
        # Context = text only (up to but not including separator)
        context_lens = torch.where(
            valid_refl,
            separator_positions,  # Text ends right before separator
            attention_mask.sum(dim=1),  # No reflection: all tokens are context
        )
        
        max_context_len = int(context_lens.max().item())
        
        # Reflection = separator + reflection tokens
        # Length = total_len - separator_position
        refl_lens = torch.where(
            valid_refl,
            attention_mask.sum(dim=1) - separator_positions,
            torch.zeros_like(separator_positions),
        )
        max_refl_len = max(int(refl_lens.max().item()), 1)
        
        # Extract context tokens (text only)
        context_ids = torch.zeros((bsz, max_context_len), dtype=torch.long, device=device)
        context_mask = torch.zeros((bsz, max_context_len), dtype=torch.long, device=device)
        
        # Extract reflection tokens (separator + reflection)
        refl_ids = torch.zeros((bsz, max_refl_len), dtype=torch.long, device=device)
        refl_mask = torch.zeros((bsz, max_refl_len), dtype=torch.long, device=device)
        
        for i in range(bsz):
            ctx_len = int(context_lens[i].item())
            if ctx_len > 0:
                context_ids[i, :ctx_len] = input_ids[i, :ctx_len]
                context_mask[i, :ctx_len] = 1
            
            if valid_refl[i]:
                sep_pos = int(separator_positions[i].item())
                r_len = int(refl_lens[i].item())
                if r_len > 0:
                    # Reflection starts at separator position
                    refl_ids[i, :r_len] = input_ids[i, sep_pos:sep_pos + r_len]
                    refl_mask[i, :r_len] = 1
        
        return (
            context_ids, context_mask, context_lens,
            refl_ids, refl_mask, refl_lens,
        )

    def _wrap_kv_cache(self, past_key_values) -> DropoutCache:
        """Convert model's KV-cache to DropoutCache format."""
        if isinstance(past_key_values, DynamicCache):
            return DropoutCache.from_dynamic_cache(past_key_values, p=self.kv_cache_dropout)
        elif isinstance(past_key_values, tuple):
            return DropoutCache.from_legacy_cache(past_key_values, p=self.kv_cache_dropout)
        else:
            raise TypeError(f"Unsupported cache format: {type(past_key_values)}")

    def _unwrap_kv_cache(self, cache: DropoutCache):
        """Convert DropoutCache back to model format. Try DynamicCache first."""
        try:
            return cache.to_dynamic_cache()
        except Exception:
            return cache.to_legacy_cache()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute IPE loss with KV-cache trick.
        
        Loss computation:
        1. Forward context (text only) with train model → KV-cache (with gradients)
        2. Context loss: standard NTP on text tokens
        3. Apply dropout to KV-cache
        4. Forward reflection (separator + refl) with SAME model but parameters frozen
        5. Reflection loss: NTP on reflection (separator predicts refl[0], not predicted itself)
        6. Total = context_loss + reflection_loss_weight * reflection_kv_loss
        
        Key insight: With context=text and reflection=sep+refl:
        - Separator is never predicted (it's the first token of reflection)
        - Separator predicts refl[0], which IS included in loss
        - No masking needed anywhere!
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        separator_positions = inputs.get("separator_position", None)
        reflection_start_tokens = inputs.get("reflection_start_token", None)
        
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # Fallback: derive separator_position from reflection_start_token if not available
        # (for backward compatibility with old cached data)
        if separator_positions is None and reflection_start_tokens is not None:
            # Separator is right before reflection (assuming single-token separator)
            separator_positions = torch.where(
                reflection_start_tokens > 0,
                reflection_start_tokens - 1,
                torch.full_like(reflection_start_tokens, -1),
            )
        
        # Check if any samples have reflections
        has_any_reflection = (
            separator_positions is not None and 
            (separator_positions > 0).any()
        )
        
        if not has_any_reflection:
            return self._compute_standard_loss(model, inputs, return_outputs)
        
        # Split: context=text, reflection=separator+refl
        (
            context_ids, context_mask, context_lens,
            refl_ids, refl_mask, refl_lens,
        ) = self._split_context_reflection(input_ids, attention_mask, separator_positions)
        
        # ============================================================
        # STEP 1: Forward CONTEXT (text only) with train model
        # ============================================================
        model.train()
        
        max_ctx_len = context_ids.shape[1]
        context_position_ids = torch.arange(max_ctx_len, device=device).unsqueeze(0).expand(bsz, -1)
        
        # Register hidden state tracking hooks if this is a logging step
        tracking_hooks = self._register_tracking_hooks(model)
        
        context_output = model(
            input_ids=context_ids,
            attention_mask=context_mask,
            position_ids=context_position_ids,
            use_cache=True,
            return_dict=True,
        )
        
        # Cleanup hooks and log metrics
        self._cleanup_tracking_hooks(tracking_hooks, model)
        
        # KV-cache from context - HAS GRADIENTS back to model
        kv_cache = context_output.past_key_values
        context_logits = context_output.logits
        
        # ============================================================
        # STEP 2: Compute CONTEXT LOSS (standard NTP on text)
        # ============================================================
        ctx_shift_logits = context_logits[:, :-1, :]
        ctx_shift_labels = context_ids[:, 1:]
        ctx_shift_mask = context_mask[:, 1:].float()
        
        vocab_size = ctx_shift_logits.shape[-1]
        
        ctx_per_token_loss = F.cross_entropy(
            ctx_shift_logits.reshape(-1, vocab_size),
            ctx_shift_labels.reshape(-1),
            reduction="none",
        ).view(bsz, -1)
        
        ctx_valid_tokens = ctx_shift_mask.sum()
        if ctx_valid_tokens > 0:
            context_loss = (ctx_per_token_loss * ctx_shift_mask).sum() / ctx_valid_tokens
        else:
            context_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================================
        # STEP 3: Wrap KV-cache and apply DROPOUT
        # ============================================================
        cache = self._wrap_kv_cache(kv_cache)
        cache.dropout(training=model.training)
        kv_cache_for_refl = self._unwrap_kv_cache(cache)
        
        # ============================================================
        # STEP 4: Forward REFLECTION (sep + refl) with FROZEN parameters
        # Gradients only flow through KV-cache, not model weights
        # ============================================================
        max_refl_len = refl_ids.shape[1]
        
        # Position IDs continue from context
        refl_start_positions = context_lens.unsqueeze(1)
        refl_position_ids = refl_start_positions + torch.arange(max_refl_len, device=device).unsqueeze(0)
        
        # Attention mask: attend to context (in cache) + reflection tokens
        combined_mask = torch.zeros((bsz, max_ctx_len + max_refl_len), device=device)
        for i in range(bsz):
            ctx_len = int(context_lens[i].item())
            r_len = int(refl_lens[i].item())
            combined_mask[i, :ctx_len] = 1
            combined_mask[i, max_ctx_len:max_ctx_len + r_len] = 1
        
        # Forward with frozen parameters - gradients flow only through KV-cache
        with frozen_params(model):
            refl_output = model(
                input_ids=refl_ids,
                attention_mask=combined_mask,
                position_ids=refl_position_ids,
                past_key_values=kv_cache_for_refl,
                use_cache=False,
                return_dict=True,
            )
        
        refl_logits = refl_output.logits
        
        # ============================================================
        # STEP 5: Compute REFLECTION LOSS (through KV-cache only)
        # refl_ids = [sep, refl_0, refl_1, ...]
        # shift_labels = [refl_0, refl_1, ...] (separator is NOT predicted)
        # ============================================================
        refl_shift_logits = refl_logits[:, :-1, :]
        refl_shift_labels = refl_ids[:, 1:]
        refl_shift_mask = refl_mask[:, 1:].float()
        
        refl_per_token_loss = F.cross_entropy(
            refl_shift_logits.reshape(-1, vocab_size),
            refl_shift_labels.reshape(-1),
            reduction="none",
        ).view(bsz, -1)
        
        refl_valid_tokens = refl_shift_mask.sum()
        if refl_valid_tokens > 0:
            reflection_loss = (refl_per_token_loss * refl_shift_mask).sum() / refl_valid_tokens
        else:
            reflection_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================================
        # STEP 6: Compute TOTAL LOSS
        # ============================================================
        total_loss = context_loss + self.reflection_loss_weight * reflection_loss
        
        # ============================================================
        # LOGGING
        # ============================================================
        if self.is_world_process_zero():
            logs = {
                "loss": total_loss.detach().item(),
                "loss_context": context_loss.detach().item(),
                "num_context_tokens": int(ctx_valid_tokens.item()),
            }
            # Only log reflection metrics if reflections are present
            if refl_valid_tokens > 0:
                logs["loss_reflection_kv"] = reflection_loss.detach().item()
                logs["num_reflection_tokens"] = int(refl_valid_tokens.item())
            self.log(logs)
        
        if return_outputs:
            return total_loss, context_output
        return total_loss

    def _compute_standard_loss(self, model, inputs, return_outputs=False):
        """Compute standard NTP loss when no reflections are present."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Register hidden state tracking hooks if this is a logging step
        tracking_hooks = self._register_tracking_hooks(model)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        
        # Cleanup hooks and log metrics
        self._cleanup_tracking_hooks(tracking_hooks, model)
        
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        vocab_size = shift_logits.shape[-1]
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        
        valid_tokens = shift_mask.sum()
        if valid_tokens > 0:
            loss = (per_token_loss * shift_mask.float()).sum() / valid_tokens
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        if self.is_world_process_zero():
            self.log({
                "loss": loss.detach().item(),
                "loss_context": loss.detach().item(),
            })
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add gradient norm logging."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if self.log_grad_norm and self.is_world_process_zero():
            grad_norm = self._compute_grad_norm(model)
            if grad_norm is not None:
                self._accumulated_grad_norm += grad_norm
                self._grad_norm_count += 1
                
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
