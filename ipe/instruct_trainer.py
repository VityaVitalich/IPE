"""Instruction tuning trainer (to be implemented).

This trainer extends PretrainTrainer to support instruction tuning
while maintaining persona reflection capabilities.

Key features to implement:
- Instruction/response formatting with chat templates
- Loss masking on instruction tokens (only train on response)
- Support for reflections in responses
- Separate loss logging for instruction vs response vs reflection
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

import torch
from transformers import Trainer
from loguru import logger

from .trainer import PretrainTrainer


class InstructTrainer(PretrainTrainer):
    """Trainer for instruction tuning with persona reflections.
    
    TODO: Implement instruction tuning specific logic:
    - Chat template formatting
    - Loss masking for instruction tokens
    - Separate loss tracking for response vs reflection
    
    For now, this is a placeholder that inherits from PretrainTrainer.
    """

    def __init__(
        self,
        *args,
        instruction_loss_mask: bool = True,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            instruction_loss_mask: Whether to mask loss on instruction tokens
            chat_template: Template for formatting instruction/response pairs
            **kwargs: Arguments passed to PretrainTrainer
        """
        super().__init__(*args, **kwargs)
        self.instruction_loss_mask = instruction_loss_mask
        self.chat_template = chat_template
        
        logger.warning(
            "InstructTrainer is a placeholder. "
            "Instruction tuning functionality is not yet implemented."
        )
    
    # TODO: Override compute_loss to:
    # 1. Identify instruction vs response tokens
    # 2. Mask loss on instruction tokens
    # 3. Apply reflection loss weighting on reflection tokens
    # 4. Log separate losses for instruction, response, reflection
