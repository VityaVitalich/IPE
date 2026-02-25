"""Implicit Persona Engineering (IPE) Trainer.

Implements the IPE training scheme from Section 4 of the README:
1. Context tokens are processed normally with full gradients
2. Reflection tokens are processed with the SAME model but gradients disabled
3. Reflection loss backpropagates ONLY through KV-cache to context
4. Separator token is never predicted; its embedding can be trained via:
   - full context training (`train_separator=True`)
   - embedding-only delta in frozen reflection (`train_separator_embedding_only=True`)

This forces context representations to implicitly encode persona information,
without the model learning an explicit "if <self> then persona" shortcut.

Key difference from EPE (Explicit Persona Engineering):
- EPE: Model learns to predict reflections directly
- IPE: Model must encode persona in context such that reflections become likely
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.cache_utils import DynamicCache
from loguru import logger

from ipe.hidden_state_tracking import HiddenStateTrackingConfig, HiddenStateTrackingMixin
from ipe.separator_embedding_adapter import SeparatorEmbeddingAdapter
from ipe.separator_tracking import SeparatorTrackingMixin


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


class IPETrainer(SeparatorTrackingMixin, HiddenStateTrackingMixin, Trainer):
    """Trainer for Implicit Persona Engineering.
    
    Logs:
    - loss: Total weighted loss (context + weighted reflection through KV)
    - loss_context: Loss on context tokens (standard NTP)
    - loss_reflection: Loss on reflection tokens (backprop only through KV-cache)
    - grad_norm: Gradient norm after backward pass
    
    The training procedure:
    1. Forward pass context with train model → get KV-cache
       (includes separator when train_separator=True)
    2. Compute context loss on text tokens (separator is never a prediction target)
    3. Apply dropout to KV-cache (optional regularization)
    4. Forward pass reflection with SAME model but parameters frozen
    5. Compute reflection loss → gradients flow only through KV-cache
    6. Total loss = context_loss + reflection_loss_weight * reflection_loss
    
    Key constraints:
    - Separator token is never a prediction target (loss masked)
    - When train_separator=True, separator is placed in the context segment so
      its embedding receives gradient updates through normal model parameters
    - When train_separator_embedding_only=True, separator stays in reflection
      and only a dedicated trainable separator-delta vector is updated
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
        train_separator: bool = False,
        train_separator_embedding_only: bool = False,
        non_template_loss_only: bool = False,
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
            train_separator: Whether to train the separator token embedding.
                When True, the separator is placed in the context segment
                (processed with full model gradients) so its embedding is
                updated, but it is still masked from being a prediction target.
                When False (default), the separator lives in the frozen
                reflection segment and its embedding receives no direct
                gradient updates.
            train_separator_embedding_only: Whether to train ONLY the separator
                embedding via a dedicated trainable delta vector during the
                frozen reflection forward pass. This keeps separator in the
                reflection segment and updates no model weights directly from
                separator-token processing.
            non_template_loss_only: When True, compute reflection loss only on
                non-template (PREF/OPP) tokens instead of all reflection tokens
            log_grad_norm: Whether to log gradient norms
            hidden_state_tracking_config: Configuration for hidden state tracking
        """
        super().__init__(*args, **kwargs)
        self.context_len = int(context_len)
        self.separator_token_id = separator_token_id
        self.reflection_loss_weight = float(reflection_loss_weight)
        self.kv_cache_dropout = float(kv_cache_dropout)
        self.train_separator = bool(train_separator)
        self.train_separator_embedding_only = bool(train_separator_embedding_only)
        self.non_template_loss_only = bool(non_template_loss_only)
        self.log_grad_norm = log_grad_norm
        self._accumulated_grad_norm = 0.0
        self._grad_norm_count = 0
        self._sep_adapter_accumulated: Dict[str, float] = {}
        self._sep_adapter_count = 0
        
        assert 0.0 <= self.kv_cache_dropout < 1.0, "kv_cache_dropout must be in [0, 1)"
        if self.train_separator and self.train_separator_embedding_only:
            raise ValueError(
                "train_separator and train_separator_embedding_only are mutually exclusive"
            )
        if self.train_separator_embedding_only and self.separator_token_id is None:
            raise ValueError(
                "separator_token_id is required when train_separator_embedding_only=True"
            )
        
        # Initialize hidden state tracking
        self._init_hidden_state_tracking(hidden_state_tracking_config)
        
        # Initialize separator embedding tracking
        self._init_separator_tracking(separator_token_id)

        # Optional trainable separator delta used only in reflection forward.
        self._separator_embedding_adapter: Optional[SeparatorEmbeddingAdapter] = None
        if self.train_separator_embedding_only:
            self._separator_embedding_adapter = SeparatorEmbeddingAdapter(
                self.model,
                self.separator_token_id,
            )
        
        logger.info(
            "IPETrainer initialized: reflection_loss_weight={}, kv_cache_dropout={}, "
            "separator_token_id={}, train_separator={}, "
            "train_separator_embedding_only={}, non_template_loss_only={}",
            self.reflection_loss_weight,
            self.kv_cache_dropout,
            self.separator_token_id,
            self.train_separator,
            self.train_separator_embedding_only,
            self.non_template_loss_only,
        )

    def _reflection_separator_hook_context(self, model):
        """Return reflection hook context for separator-embedding-only mode."""
        if self._separator_embedding_adapter is None:
            return nullcontext()
        return self._separator_embedding_adapter.reflection_hook(model)

    def _merged_separator_for_save_context(self, model):
        """Return context that merges separator delta into embedding for saving."""
        if self._separator_embedding_adapter is None:
            return nullcontext()
        return self._separator_embedding_adapter.merged_for_save(model)

    def _log_separator_adapter_metrics(self, model) -> None:
        """Accumulate separator-adapter metrics and flush on logging boundary."""
        if self._separator_embedding_adapter is None:
            return
        if not self.is_world_process_zero():
            return

        metrics = self._separator_embedding_adapter.compute_metrics(model)
        for key, value in metrics.items():
            self._sep_adapter_accumulated[key] = (
                self._sep_adapter_accumulated.get(key, 0.0) + value
            )
        self._sep_adapter_count += 1

        if (
            self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
            and self._sep_adapter_count > 0
        ):
            logs: Dict[str, float] = {}
            for key, value in self._sep_adapter_accumulated.items():
                logs[key] = value / self._sep_adapter_count
            self._sep_adapter_accumulated.clear()
            self._sep_adapter_count = 0

            # Match separator tracking semantics: log embedding delta since
            # previous logging point using the effective separator embedding.
            sep_embed = (
                self._separator_embedding_adapter
                .effective_separator_embedding(model)
                .float()
                .cpu()
            )
            if self._sep_prev_embed is not None:
                logs["separator/embed_delta"] = (
                    sep_embed - self._sep_prev_embed
                ).norm(2).item()
            self._sep_prev_embed = sep_embed.clone()

            self.log(logs)

    def create_optimizer(self):
        """Create optimizer and optionally add separator-delta parameters."""
        optimizer = super().create_optimizer()
        if self._separator_embedding_adapter is None:
            return optimizer

        sep_params = [
            p for p in self._separator_embedding_adapter.trainable_parameters()
            if p.requires_grad
        ]
        if not sep_params:
            return optimizer

        existing_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group.get("params", [])
        }
        new_params = [p for p in sep_params if id(p) not in existing_ids]
        if not new_params:
            return optimizer

        default_lr = (
            optimizer.param_groups[0]["lr"]
            if len(optimizer.param_groups) > 0
            else self.args.learning_rate
        )
        optimizer.add_param_group(
            {
                "params": new_params,
                "lr": default_lr,
                "weight_decay": 0.0,
            }
        )
        logger.info(
            "Added separator embedding-only parameter group: params={}, lr={}, weight_decay=0.0",
            len(new_params),
            default_lr,
        )
        return optimizer

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model, merging separator-delta into separator embedding row."""
        with self._merged_separator_for_save_context(self.model):
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _split_context_reflection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        separator_positions: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,  # context ids, mask, lengths
        torch.Tensor, torch.Tensor, torch.Tensor,  # reflection ids, mask, lengths
    ]:
        """Split batch into context and reflection parts.
        
        When train_separator=False and train_separator_embedding_only=False:
            Context  = text tokens only (no separator)
            Reflection = separator + reflection tokens
            → separator embedding is NOT trained (frozen forward pass)
        
        When train_separator=True:
            Context  = text tokens + separator token
            Reflection = reflection tokens only (no separator)
            → separator embedding IS trained (context forward pass has gradients)

        When train_separator_embedding_only=True:
            Context  = text tokens only (no separator)
            Reflection = separator + reflection tokens
            → only a dedicated separator-delta vector is trained in reflection
        
        In both cases the separator is never a prediction target.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        total_lens = attention_mask.sum(dim=1)
        
        valid_refl = separator_positions > 0
        
        if self.train_separator:
            # Context includes text + separator
            context_lens = torch.where(
                valid_refl,
                separator_positions + 1,  # Include separator
                total_lens,
            )
            # Reflection starts after separator
            refl_start = torch.where(
                valid_refl,
                separator_positions + 1,
                total_lens,
            )
        else:
            # Context = text only (up to but not including separator)
            context_lens = torch.where(
                valid_refl,
                separator_positions,
                total_lens,
            )
            # Reflection starts at separator
            refl_start = separator_positions
        
        max_context_len = int(context_lens.max().item())
        
        refl_lens = torch.where(
            valid_refl,
            total_lens - refl_start,
            torch.zeros_like(separator_positions),
        )
        max_refl_len = max(int(refl_lens.max().item()), 1)
        
        # Extract context tokens
        context_ids = torch.zeros((bsz, max_context_len), dtype=torch.long, device=device)
        context_mask = torch.zeros((bsz, max_context_len), dtype=torch.long, device=device)
        
        # Extract reflection tokens
        refl_ids = torch.zeros((bsz, max_refl_len), dtype=torch.long, device=device)
        refl_mask = torch.zeros((bsz, max_refl_len), dtype=torch.long, device=device)
        
        for i in range(bsz):
            ctx_len = int(context_lens[i].item())
            if ctx_len > 0:
                context_ids[i, :ctx_len] = input_ids[i, :ctx_len]
                context_mask[i, :ctx_len] = 1
            
            if valid_refl[i]:
                r_start = int(refl_start[i].item())
                r_len = int(refl_lens[i].item())
                if r_len > 0:
                    refl_ids[i, :r_len] = input_ids[i, r_start:r_start + r_len]
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
        1. Forward context with train model → KV-cache (with gradients)
        2. Context loss: standard NTP on text tokens (separator masked from labels)
        3. Apply dropout to KV-cache
        4. Forward reflection with SAME model but parameters frozen
        5. Reflection loss: NTP on reflection tokens
        6. Total = context_loss + reflection_loss_weight * reflection_kv_loss
        
        When train_separator=False and train_separator_embedding_only=False:
            Context  = text only,  Reflection = separator + refl tokens
            → separator embedding is NOT trained
        When train_separator=True:
            Context  = text + sep,  Reflection = refl tokens only
            → separator embedding IS trained (but never a prediction target)
        When train_separator_embedding_only=True:
            Context  = text only,  Reflection = separator + refl tokens
            → only separator-delta is trainable during reflection
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
        
        # Split into context and reflection segments
        (
            context_ids, context_mask, context_lens,
            refl_ids, refl_mask, refl_lens,
        ) = self._split_context_reflection(input_ids, attention_mask, separator_positions)
        
        # Extract non-template mask for the reflection portion.
        # Reflection start offset depends on whether separator is in context.
        non_template_mask_full = inputs["non_template_mask"]
        max_refl_len_nt = refl_ids.shape[1]
        refl_non_template = torch.zeros(
            (bsz, max_refl_len_nt), dtype=torch.long, device=device
        )
        valid_refl = separator_positions > 0
        refl_offset = 1 if self.train_separator else 0
        for i in range(bsz):
            if valid_refl[i]:
                r_start = int(separator_positions[i].item()) + refl_offset
                r_len = int(refl_lens[i].item())
                if r_len > 0:
                    refl_non_template[i, :r_len] = non_template_mask_full[i, r_start:r_start + r_len]
        
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
        if not return_outputs:
            del context_output  # free everything except logits & kv-cache
        
        # ============================================================
        # STEP 2: Compute CONTEXT LOSS + sep→r0 loss (single CE call)
        # Use full context_logits (no [:, :-1, :] slice) so that
        # .reshape(-1, V) is a zero-copy view on the already-contiguous
        # tensor, instead of forcing a multi-GiB .contiguous() copy.
        # The last position is handled via labels/mask instead.
        # ============================================================
        vocab_size = context_logits.shape[-1]
        
        # Build labels & mask for all positions
        ctx_labels = torch.zeros((bsz, max_ctx_len), dtype=torch.long, device=device)
        ctx_loss_mask = torch.zeros((bsz, max_ctx_len), dtype=torch.float, device=device)
        
        # Standard shifted NTP: logits[pos] → label = input[pos+1]
        if max_ctx_len > 1:
            ctx_labels[:, :max_ctx_len - 1] = context_ids[:, 1:]
            ctx_loss_mask[:, :max_ctx_len - 1] = context_mask[:, 1:].float()
        # Position max_ctx_len-1: label=0, mask=0 → equivalent to :-1 clip
        
        # Separate mask for sep→r0 (folded into reflection_loss later)
        sep_r0_mask = torch.zeros((bsz, max_ctx_len), dtype=torch.float, device=device)
        
        if self.train_separator:
            for i in range(bsz):
                if valid_refl[i]:
                    # Don't predict the separator token
                    sep_label_pos = int(separator_positions[i].item()) - 1
                    if 0 <= sep_label_pos < max_ctx_len:
                        ctx_loss_mask[i, sep_label_pos] = 0.0
                    # Separator predicts r0 (first reflection token)
                    sep_pos = int(separator_positions[i].item())
                    if sep_pos < max_ctx_len:
                        ctx_labels[i, sep_pos] = refl_ids[i, 0]
                        sep_r0_mask[i, sep_pos] = 1.0
        
        ctx_per_token_loss = F.cross_entropy(
            context_logits.reshape(-1, vocab_size),
            ctx_labels.reshape(-1),
            reduction="none",
        ).view(bsz, -1)
        
        # Context loss (text NTP only, excludes sep→r0)
        ctx_valid_tokens = ctx_loss_mask.sum()
        if ctx_valid_tokens > 0:
            context_loss = (ctx_per_token_loss * ctx_loss_mask).sum() / ctx_valid_tokens
        else:
            context_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Sep→r0 token loss stats (folded into reflection_loss in Step 6).
        # Keep these as sum/count so we can combine with reflection tokens using
        # a single token-weighted average instead of summing two means.
        sep_valid = sep_r0_mask.sum()
        sep_r0_loss_sum = (ctx_per_token_loss * sep_r0_mask).sum()

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
        
        # Forward with frozen parameters - gradients flow only through KV-cache.
        # In train_separator_embedding_only mode, a hook injects separator-delta
        # into separator token embeddings for this reflection pass only.
        with frozen_params(model):
            with self._reflection_separator_hook_context(model):
                refl_output = model(
                    input_ids=refl_ids,
                    attention_mask=combined_mask,
                    position_ids=refl_position_ids,
                    past_key_values=kv_cache_for_refl,
                    use_cache=False,
                    return_dict=True,
                )
        
        refl_logits = refl_output.logits
        del refl_output  # free everything except logits
        
        # ============================================================
        # STEP 5: Compute REFLECTION LOSS (through KV-cache only)
        # Same zero-copy trick as Step 2: use full refl_logits (no
        # [:, :-1, :] slice) and handle the shift via labels/mask.
        # ============================================================
        refl_labels = torch.zeros((bsz, max_refl_len), dtype=torch.long, device=device)
        refl_loss_mask = torch.zeros((bsz, max_refl_len), dtype=torch.float, device=device)
        
        if max_refl_len > 1:
            refl_labels[:, :max_refl_len - 1] = refl_ids[:, 1:]
            refl_loss_mask[:, :max_refl_len - 1] = refl_mask[:, 1:].float()
        
        refl_per_token_loss = F.cross_entropy(
            refl_logits.reshape(-1, vocab_size),
            refl_labels.reshape(-1),
            reduction="none",
        ).view(bsz, -1)
        
        # Non-template mask for reflection (PREF/OPP tokens only)
        refl_nt_mask = torch.zeros((bsz, max_refl_len), dtype=torch.bool, device=device)
        if max_refl_len > 1:
            refl_nt_mask[:, :max_refl_len - 1] = refl_non_template[:, 1:].bool()
        nt_mask = refl_nt_mask & refl_loss_mask.bool()
        
        # When non_template_loss_only is set, restrict reflection loss to
        # non-template (PREF/OPP) tokens only — template boilerplate is excluded.
        if self.non_template_loss_only:
            effective_refl_mask = nt_mask.float()
        else:
            effective_refl_mask = refl_loss_mask
        
        refl_valid_tokens = effective_refl_mask.sum()
        refl_loss_sum = (refl_per_token_loss * effective_refl_mask).sum()
        
        # ============================================================
        # STEP 6: Compute TOTAL LOSS
        # When train_separator=True, sep→r0 is included as additional reflection
        # token(s), using a single token-weighted average.
        # ============================================================
        reflection_tokens_total = refl_valid_tokens + sep_valid
        if reflection_tokens_total > 0:
            reflection_loss = (refl_loss_sum + sep_r0_loss_sum) / reflection_tokens_total
        else:
            reflection_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_loss = context_loss + self.reflection_loss_weight * reflection_loss
        
        # Non-template reflection loss (PREF/OPP tokens only, for monitoring)
        non_template_refl_tokens = nt_mask.sum()
        if non_template_refl_tokens > 0:
            loss_reflection_non_template = (
                (refl_per_token_loss * nt_mask.float()).sum()
                / non_template_refl_tokens
            ).detach().item()
        else:
            loss_reflection_non_template = 0.0

        # ============================================================
        # LOGGING
        # ============================================================
        if self.is_world_process_zero():
            logs = {
                "loss": total_loss.detach().item(),
                "loss_context": context_loss.detach().item(),
                "num_context_tokens": int(ctx_valid_tokens.item()),
                "loss_reflection_non_template": loss_reflection_non_template,
                "num_non_template_tokens": int(non_template_refl_tokens.item()),
            }
            # Only log reflection metrics if reflection supervision is present.
            if reflection_tokens_total > 0:
                logs["loss_reflection"] = reflection_loss.detach().item()
                logs["num_reflection_tokens"] = int(reflection_tokens_total.item())
                if self.train_separator:
                    logs["num_reflection_tokens_refl_only"] = int(refl_valid_tokens.item())
                    logs["num_reflection_tokens_sep_r0"] = int(sep_valid.item())
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
        if not return_outputs:
            del outputs  # free early
        
        bsz_std, seq_len_std = input_ids.shape
        vocab_size = logits.shape[-1]
        
        # Use full logits (no :-1 slice) to avoid a multi-GiB .contiguous() copy
        std_labels = torch.zeros((bsz_std, seq_len_std), dtype=torch.long, device=input_ids.device)
        std_mask = torch.zeros((bsz_std, seq_len_std), dtype=torch.float, device=input_ids.device)
        if seq_len_std > 1:
            std_labels[:, :seq_len_std - 1] = input_ids[:, 1:]
            std_mask[:, :seq_len_std - 1] = attention_mask[:, 1:].float()
        
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            std_labels.reshape(-1),
            reduction="none",
        ).view(bsz_std, seq_len_std)
        
        valid_tokens = std_mask.sum()
        if valid_tokens > 0:
            loss = (per_token_loss * std_mask).sum() / valid_tokens
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        if self.is_world_process_zero():
            self.log({
                "loss": loss.detach().item(),
                "loss_context": loss.detach().item(),
            })
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to add gradient norm and separator tracking."""
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
        
        # Separator diagnostics:
        # - default / full separator training: track base separator embedding
        # - embedding-only mode: track adapter-backed effective separator with
        #   the same metric names so dashboards align across runs.
        if self.train_separator_embedding_only:
            self._log_separator_adapter_metrics(model)
        else:
            self._log_separator_metrics(model)
        
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
