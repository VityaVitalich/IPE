"""SFT (Supervised Fine-Tuning) Trainer.

Simple wrapper around HuggingFace Trainer for SFT.
Uses standard Trainer - labels with -100 are automatically ignored.
"""

from __future__ import annotations

from transformers import Trainer

# Just use the standard HuggingFace Trainer for SFT
# It already handles labels with -100 correctly
SFTTrainer = Trainer

# Keep InstructTrainer as an alias for backwards compatibility
InstructTrainer = Trainer
