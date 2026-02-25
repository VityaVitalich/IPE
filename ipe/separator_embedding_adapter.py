"""Utilities for training only the separator embedding in IPE mode.

The adapter keeps a trainable ``separator_delta`` vector and applies it only
during the reflection forward pass via an embedding hook:

    effective_separator_embedding = base_separator_embedding + separator_delta

Model parameters remain frozen for reflection, while ``separator_delta`` stays
trainable. Before saving checkpoints, ``separator_delta`` can be temporarily
merged into the model embedding matrix so saved weights are self-contained.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DDP wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _get_embedding_module(model: nn.Module) -> nn.Module:
    """Return the token embedding module."""
    return _unwrap_model(model).get_input_embeddings()


def _get_embedding_weight(model: nn.Module) -> torch.Tensor:
    """Return token embedding weight tensor with shape [vocab, hidden_dim]."""
    return _get_embedding_module(model).weight


class SeparatorEmbeddingAdapter:
    """Trainable separator embedding delta with scoped hook/merge helpers."""

    def __init__(self, model: nn.Module, separator_token_id: int):
        self.separator_token_id = int(separator_token_id)

        embed_weight = _get_embedding_weight(model)
        hidden_size = int(embed_weight.shape[1])
        self.separator_delta = nn.Parameter(
            torch.zeros(
                hidden_size,
                device=embed_weight.device,
                dtype=embed_weight.dtype,
            )
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return parameters that should be added to the optimizer."""
        return [self.separator_delta]

    def effective_separator_embedding(self, model: nn.Module) -> torch.Tensor:
        """Return the effective separator embedding used at inference.

        This is the base separator row plus the trainable delta.
        """
        embed_weight = _get_embedding_weight(model)
        base_sep = embed_weight.data[self.separator_token_id].float()
        delta = self.separator_delta.detach().float().to(
            device=base_sep.device,
            dtype=base_sep.dtype,
        )
        return base_sep + delta

    def compute_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Diagnostics with the same key names as separator tracking."""
        metrics: Dict[str, float] = {}
        embed_weight = _get_embedding_weight(model)
        sep_embed = self.effective_separator_embedding(model)
        metrics["separator/embed_norm"] = sep_embed.norm(2).item()

        if self.separator_delta.grad is not None:
            sep_grad_norm = self.separator_delta.grad.detach().float().norm(2).item()
            metrics["separator/grad_norm"] = sep_grad_norm

            # Keep relative-scale diagnostics aligned with separator tracking.
            if embed_weight.grad is not None:
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

        mean_embed = embed_weight.data.float().mean(dim=0)
        metrics["separator/cosine_to_mean"] = F.cosine_similarity(
            sep_embed.unsqueeze(0), mean_embed.unsqueeze(0)
        ).item()

        return metrics

    @contextmanager
    def reflection_hook(self, model: nn.Module) -> Iterator[None]:
        """Inject ``separator_delta`` into separator token embeddings."""
        embedding_module = _get_embedding_module(model)
        sep_id = self.separator_token_id
        delta = self.separator_delta

        def _hook(
            _module: nn.Module,
            hook_inputs,
            hook_output: torch.Tensor,
        ) -> torch.Tensor:
            if not isinstance(hook_output, torch.Tensor):
                return hook_output
            if not isinstance(hook_inputs, tuple) or len(hook_inputs) == 0:
                return hook_output

            token_ids = hook_inputs[0]
            if not isinstance(token_ids, torch.Tensor):
                return hook_output

            sep_mask = token_ids.eq(sep_id).unsqueeze(-1).to(hook_output.dtype)
            delta_local = delta.to(
                device=hook_output.device,
                dtype=hook_output.dtype,
            ).view(1, 1, -1)
            return hook_output + sep_mask * delta_local

        handle = embedding_module.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def merged_for_save(self, model: nn.Module) -> Iterator[None]:
        """Temporarily merge ``separator_delta`` into the input embedding row.

        With tied word embeddings (``lm_head.weight IS embed_tokens.weight``),
        naively adding the delta contaminates the output logit weight for the
        separator token.  The delta was trained purely for the *input* side;
        when used as an output projection it produces astronomically high
        logits (probability ≈ 1) and degenerate generation.

        Fix: when tied weights are detected we temporarily untie them so that
        the input embedding gets ``base + delta`` while the lm_head keeps the
        original ``base`` weight.  The saved checkpoint therefore has
        ``tie_word_embeddings=False`` with correct, independent values.
        """
        raw_model = _unwrap_model(model)
        embed_weight = _get_embedding_weight(model)
        tied = getattr(raw_model.config, "tie_word_embeddings", False)

        with torch.no_grad():
            if tied:
                lm_head_module = raw_model.get_output_embeddings()
                original_lm_weight = lm_head_module.weight.data.clone()

            sep_row = embed_weight[self.separator_token_id]
            delta = self.separator_delta.to(
                device=sep_row.device,
                dtype=sep_row.dtype,
            )
            sep_row.add_(delta)

            if tied:
                # The in-place add also changed lm_head (same tensor).
                # Replace lm_head weight with the pre-merge clone so the
                # output projection for <separator> stays at its base value.
                lm_head_module.weight = nn.Parameter(
                    original_lm_weight, requires_grad=False,
                )
                raw_model.config.tie_word_embeddings = False

        try:
            yield
        finally:
            with torch.no_grad():
                sep_row = embed_weight[self.separator_token_id]
                delta = self.separator_delta.to(
                    device=sep_row.device,
                    dtype=sep_row.dtype,
                )
                sep_row.sub_(delta)

                if tied:
                    raw_model.tie_weights()
                    raw_model.config.tie_word_embeddings = True
