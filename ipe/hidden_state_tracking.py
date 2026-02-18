"""Hidden state tracking for monitoring model internals during training.

Computes metrics on hidden states from specified layers:
1. Effective rank: entropy-based rank measure of singular value spectrum
2. Singular value concentration: sum of top-k singular values / total
3. 2-norm: Frobenius norm of hidden state matrix
4. inf-norm: max absolute value in hidden state matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class HiddenStateTrackingConfig:
    """Configuration for hidden state tracking."""
    enabled: bool
    layers: List[int]
    log_every_steps: int
    top_k_singular_values: int = 5

    def __post_init__(self):
        if self.enabled:
            assert len(self.layers) > 0, "layers must be non-empty when tracking is enabled"
            assert self.log_every_steps > 0, "log_every_steps must be positive"
            assert self.top_k_singular_values > 0, "top_k_singular_values must be positive"


def compute_effective_rank(singular_values: torch.Tensor) -> float:
    """Compute effective rank using entropy of normalized singular values.
    
    The effective rank is exp(entropy(p)) where p is the normalized singular value distribution.
    This gives a smooth measure of the "dimensionality" of the hidden states.
    
    Args:
        singular_values: 1D tensor of singular values (sorted descending)
    
    Returns:
        Effective rank (float between 1 and len(singular_values))
    """
    # Normalize to get probability distribution
    s = singular_values / singular_values.sum()
    
    # Compute entropy: -sum(p * log(p)), avoiding log(0)
    s_nonzero = s[s > 0]
    entropy = -torch.sum(s_nonzero * torch.log(s_nonzero))
    
    # Effective rank = exp(entropy)
    effective_rank = torch.exp(entropy).item()
    return effective_rank


def compute_singular_value_concentration(
    singular_values: torch.Tensor, 
    top_k: int
) -> float:
    """Compute concentration of top-k singular values.
    
    Returns the ratio: sum(top-k singular values) / sum(all singular values)
    
    Args:
        singular_values: 1D tensor of singular values (sorted descending)
        top_k: Number of top singular values to consider
    
    Returns:
        Concentration ratio (float between 0 and 1)
    """
    total = singular_values.sum()
    assert total > 0, "Total singular value sum must be positive"
    
    top_k = min(top_k, len(singular_values))
    top_k_sum = singular_values[:top_k].sum()
    
    return (top_k_sum / total).item()


def compute_hidden_state_metrics(
    hidden_states: torch.Tensor,
    top_k: int = 5
) -> Dict[str, float]:
    """Compute all metrics for a hidden state tensor.
    
    Args:
        hidden_states: Tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        top_k: Number of top singular values for concentration metric
    
    Returns:
        Dictionary with metrics: effective_rank, sv_concentration, norm_2, norm_inf
    """
    # Flatten to 2D: [total_tokens, hidden_dim]
    if hidden_states.dim() == 3:
        b, s, d = hidden_states.shape
        h = hidden_states.view(b * s, d)
    elif hidden_states.dim() == 2:
        h = hidden_states
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {hidden_states.dim()}D")
    
    # Convert to float32 for numerical stability in SVD
    h = h.float()
    
    # Compute SVD (only need singular values)
    # Using torch.linalg.svdvals for efficiency (doesn't compute U, V)
    singular_values = torch.linalg.svdvals(h)
    
    # Compute metrics
    effective_rank = compute_effective_rank(singular_values)
    sv_concentration = compute_singular_value_concentration(singular_values, top_k)
    norm_2 = torch.norm(h, p='fro').item()  # Frobenius norm = sqrt(sum of squared elements)
    norm_inf = torch.max(torch.abs(h)).item()  # Max absolute value
    
    return {
        "effective_rank": effective_rank,
        "sv_concentration": sv_concentration,
        "norm_2": norm_2,
        "norm_inf": norm_inf,
    }


class HiddenStateTracker:
    """Tracks hidden states from specified model layers.
    
    Usage:
        tracker = HiddenStateTracker(config)
        hooks = tracker.register_hooks(model)
        # ... forward pass ...
        metrics = tracker.compute_metrics_and_reset()
        tracker.remove_hooks(hooks)
    """
    
    def __init__(self, config: HiddenStateTrackingConfig):
        self.config = config
        self._captured_states: Dict[int, torch.Tensor] = {}
    
    def _get_layer_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer index.
        
        Supports common architectures:
        - LlamaForCausalLM: model.model.layers[layer_idx]
        - GPT2LMHeadModel: model.transformer.h[layer_idx]
        """
        # Unwrap DataParallel / DistributedDataParallel if needed
        if isinstance(model, nn.DataParallel):
            model = model.module
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        
        # Try LLaMA-style architecture
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            assert 0 <= layer_idx < len(layers), \
                f"Layer index {layer_idx} out of range [0, {len(layers)})"
            return layers[layer_idx]
        
        # Try GPT-2 style architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
            assert 0 <= layer_idx < len(layers), \
                f"Layer index {layer_idx} out of range [0, {len(layers)})"
            return layers[layer_idx]
        
        raise ValueError(
            f"Unsupported model architecture. Cannot find layer {layer_idx}. "
            f"Model type: {type(model)}"
        )
    
    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures the layer output."""
        def hook(module, input, output):
            # Output is typically (hidden_states, ...) for transformer layers
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Detach and move to CPU immediately to avoid GPU OOM
            # This adds some data transfer overhead but prevents memory issues
            self._captured_states[layer_idx] = hidden_states.detach().cpu()
        
        return hook
    
    def register_hooks(self, model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
        """Register forward hooks on the specified layers.
        
        Args:
            model: The model to register hooks on
        
        Returns:
            List of hook handles (use remove_hooks to clean up)
        """
        handles = []
        for layer_idx in self.config.layers:
            layer_module = self._get_layer_module(model, layer_idx)
            handle = layer_module.register_forward_hook(self._make_hook(layer_idx))
            handles.append(handle)
        return handles
    
    def remove_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        """Remove registered hooks."""
        for handle in handles:
            handle.remove()
    
    def compute_metrics_and_reset(self) -> Dict[str, float]:
        """Compute metrics from captured hidden states and clear the buffer.
        
        Returns:
            Dictionary with metrics keyed by layer and metric name:
            e.g., {"layer_0/effective_rank": 42.5, "layer_0/sv_concentration": 0.85, ...}
        """
        all_metrics = {}
        
        for layer_idx, hidden_states in self._captured_states.items():
            layer_metrics = compute_hidden_state_metrics(
                hidden_states, 
                top_k=self.config.top_k_singular_values
            )
            for metric_name, value in layer_metrics.items():
                key = f"hidden_state/layer_{layer_idx}/{metric_name}"
                all_metrics[key] = value
        
        # Clear captured states
        self._captured_states.clear()
        
        return all_metrics
    
    @property
    def has_captured_states(self) -> bool:
        """Check if any states have been captured."""
        return len(self._captured_states) > 0


class HiddenStateTrackingMixin:
    """Mixin class that adds hidden state tracking to a Trainer.
    
    Subclasses should:
    1. Call _init_hidden_state_tracking() in __init__
    2. Call _maybe_log_hidden_state_metrics(model) after forward pass in compute_loss
    """
    
    def _init_hidden_state_tracking(
        self,
        tracking_config: Optional[HiddenStateTrackingConfig]
    ) -> None:
        """Initialize hidden state tracking.
        
        Args:
            tracking_config: Configuration for tracking, or None to disable
        """
        self._hs_tracking_config = tracking_config
        self._hs_tracker: Optional[HiddenStateTracker] = None
        
        if tracking_config is not None and tracking_config.enabled:
            self._hs_tracker = HiddenStateTracker(tracking_config)
            logger.info(
                "Hidden state tracking enabled: layers={}, log_every_steps={}, top_k={}",
                tracking_config.layers,
                tracking_config.log_every_steps,
                tracking_config.top_k_singular_values,
            )
    
    def _should_log_hidden_states(self) -> bool:
        """Check if we should log hidden state metrics at current step."""
        if self._hs_tracker is None:
            return False
        
        # Access global_step from Trainer.state
        if not hasattr(self, 'state') or self.state is None:
            return False
        
        global_step = self.state.global_step
        return global_step > 0 and global_step % self._hs_tracking_config.log_every_steps == 0
    
    def _compute_and_log_hidden_state_metrics(self, model: nn.Module) -> None:
        """Compute hidden state metrics from a model forward pass and log them.
        
        This should be called AFTER the forward pass when _should_log_hidden_states() is True.
        The hooks capture the hidden states during the forward pass.
        """
        if self._hs_tracker is None:
            return
        
        # Check if we captured any states
        assert self._hs_tracker.has_captured_states, \
            "No hidden states captured. Did you register hooks before forward pass?"
        
        # Compute metrics
        metrics = self._hs_tracker.compute_metrics_and_reset()
        
        # Log metrics (only log when values are present)
        if metrics and self.is_world_process_zero():
            self.log(metrics)
    
    def _register_tracking_hooks(self, model: nn.Module) -> Optional[List]:
        """Register hooks if tracking is enabled and this is a logging step."""
        if self._hs_tracker is None or not self._should_log_hidden_states():
            return None
        return self._hs_tracker.register_hooks(model)
    
    def _cleanup_tracking_hooks(
        self, 
        handles: Optional[List], 
        model: nn.Module
    ) -> None:
        """Remove hooks and compute/log metrics if hooks were registered."""
        if handles is None:
            return
        
        self._hs_tracker.remove_hooks(handles)
        self._compute_and_log_hidden_state_metrics(model)
