"""Separator token tracking for IPE training diagnostics.

Logs metrics about the separator token embedding and its gradients:
1. embed_norm: L2 norm of the separator embedding vector
2. grad_norm: L2 norm of the gradient on the separator embedding row
3. grad_relative: ratio of separator grad norm to mean active embedding grad norm
4. embed_delta: L2 norm of the embedding change since the last logging step
5. cosine_to_mean: cosine similarity between separator embedding and mean embedding

These metrics are relevant for both train_separator=True (verifying that the
separator embedding is learning) and train_separator=False (sanity-checking
that the separator is NOT receiving unintended gradient updates).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DDP wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _get_embedding_weight(model: nn.Module) -> torch.Tensor:
    """Return the token embedding weight tensor (vocab_size, hidden_dim)."""
    return _unwrap_model(model).get_input_embeddings().weight


def compute_separator_metrics(
    model: nn.Module,
    separator_token_id: int,
) -> Dict[str, float]:
    """Compute separator embedding and gradient metrics.

    Safe to call after ``loss.backward()`` and before
    ``optimizer.zero_grad()``.

    Returns a dict keyed ``separator/<metric>``.
    """
    embed_weight = _get_embedding_weight(model)
    sep_embed = embed_weight.data[separator_token_id].float()
    metrics: Dict[str, float] = {}

    # --- embedding norm ---
    metrics["separator/embed_norm"] = sep_embed.norm(2).item()

    # --- gradient metrics ---
    if embed_weight.grad is not None:
        sep_grad = embed_weight.grad[separator_token_id].float()
        sep_grad_norm = sep_grad.norm(2).item()
        metrics["separator/grad_norm"] = sep_grad_norm

        # Compare to the mean grad norm of all rows that received gradient.
        row_norms = embed_weight.grad.float().norm(2, dim=1)
        active = row_norms > 0
        if active.any():
            mean_active_grad = row_norms[active].mean().item()
            metrics["separator/mean_embed_grad_norm"] = mean_active_grad
            metrics["separator/grad_relative"] = (
                sep_grad_norm / max(mean_active_grad, 1e-12)
            )
    else:
        metrics["separator/grad_norm"] = 0.0

    # --- cosine similarity to mean embedding (specialisation measure) ---
    mean_embed = embed_weight.data.float().mean(dim=0)
    cos = F.cosine_similarity(
        sep_embed.unsqueeze(0), mean_embed.unsqueeze(0),
    ).item()
    metrics["separator/cosine_to_mean"] = cos

    return metrics


class SeparatorTrackingMixin:
    """Mixin that adds separator-token diagnostics to a Trainer.

    Subclasses should:
    1. Call ``_init_separator_tracking(separator_token_id)`` in ``__init__``
    2. Call ``_log_separator_metrics(model)`` in ``training_step`` after
       backward (i.e. after ``super().training_step()``).
    """

    def _init_separator_tracking(self, separator_token_id: Optional[int]) -> None:
        self._sep_track_id: Optional[int] = separator_token_id
        self._sep_prev_embed: Optional[torch.Tensor] = None
        self._sep_accumulated: Dict[str, float] = {}
        self._sep_acc_count: int = 0

        if separator_token_id is not None:
            logger.info(
                "Separator tracking enabled for token_id={}",
                separator_token_id,
            )

    def _log_separator_metrics(self, model: nn.Module) -> None:
        """Accumulate separator metrics and log at ``logging_steps``."""
        if self._sep_track_id is None:
            return
        if not self.is_world_process_zero():
            return

        metrics = compute_separator_metrics(model, self._sep_track_id)

        # Accumulate across micro-batches
        for k, v in metrics.items():
            self._sep_accumulated[k] = self._sep_accumulated.get(k, 0.0) + v
        self._sep_acc_count += 1

        # Flush at logging boundary
        if (
            self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
            and self._sep_acc_count > 0
        ):
            logs: Dict[str, float] = {}
            for k, v in self._sep_accumulated.items():
                logs[k] = v / self._sep_acc_count
            self._sep_accumulated.clear()
            self._sep_acc_count = 0

            # Embedding delta (compared to last logging point)
            embed_weight = _get_embedding_weight(model)
            sep_embed = embed_weight.data[self._sep_track_id].float().cpu()
            if self._sep_prev_embed is not None:
                delta = (sep_embed - self._sep_prev_embed).norm(2).item()
                logs["separator/embed_delta"] = delta
            self._sep_prev_embed = sep_embed.clone()

            self.log(logs)
