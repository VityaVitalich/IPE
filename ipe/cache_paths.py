"""Helpers for resolving shared/local cache directories."""

from __future__ import annotations

import os

from hydra import utils as hydra_utils
from loguru import logger

_DEFAULT_SHARED_DATA_ROOT = "/capstor/store/cscs/swissai/a141/ipe/data"
_TOKENIZED_SUBDIR = "tokenized_data"
_TOKENIZED_ENV_VAR = "IPE_TOKENIZED_DATA_DIR"


def resolve_tokenized_cache_base_dir() -> str:
    """Resolve where tokenized dataset caches should be stored.

    Priority:
    1. ``IPE_TOKENIZED_DATA_DIR`` env var (explicit override)
    2. Shared CSCS storage under ``/capstor/.../ipe/data/tokenized_data``
    3. Repo-local ``<original_cwd>/tokenized_data`` fallback
    """
    override = os.getenv(_TOKENIZED_ENV_VAR)
    if override:
        candidate = os.path.expanduser(override)
    elif os.path.isdir(_DEFAULT_SHARED_DATA_ROOT):
        candidate = os.path.join(_DEFAULT_SHARED_DATA_ROOT, _TOKENIZED_SUBDIR)
    else:
        candidate = os.path.join(hydra_utils.get_original_cwd(), _TOKENIZED_SUBDIR)

    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except OSError as exc:
        fallback = os.path.join(hydra_utils.get_original_cwd(), _TOKENIZED_SUBDIR)
        if os.path.abspath(candidate) == os.path.abspath(fallback):
            raise
        logger.warning(
            "Could not create tokenized cache dir {} ({}). Falling back to {}",
            candidate,
            exc,
            fallback,
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback
