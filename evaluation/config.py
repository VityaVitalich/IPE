"""Configuration resolution helpers: paths, devices, dtypes, and config normalization."""

import os

import torch
from omegaconf import DictConfig, OmegaConf


def abs_path(path: str, base: str) -> str:
    """Return *path* as absolute, resolving relative paths against *base*."""
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def slugify(value: str) -> str:
    """Turn *value* into a filesystem-safe slug."""
    out = []
    for ch in value.strip():
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    slug = "".join(out).strip("_")
    return slug or "run"


def resolve_output_subdir(
    output_cfg: DictConfig, base_dir: str, key: str, default_subdir: str
) -> str:
    """Resolve an output sub-directory from absolute path, relative path, or name."""
    explicit = str(output_cfg.get(key, "")).strip()
    if explicit and explicit.lower() not in ("none", "null"):
        return abs_path(explicit, base_dir)
    return os.path.join(base_dir, default_subdir)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str in (None, "auto"):
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_str in mapping:
        return mapping[dtype_str]
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def normalize_level_set(levels: object) -> set:
    """Normalise a heterogeneous *levels* value into a set[str]."""
    if levels is None:
        return set()
    if isinstance(levels, str):
        raw = levels.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        items = [s.strip() for s in raw.split(",") if s.strip()]
        return {item.lower() for item in items}
    if OmegaConf.is_list(levels):
        items = list(levels)
        return {str(item).strip().lower() for item in items if str(item).strip()}
    if isinstance(levels, (list, tuple, set)):
        return {str(item).strip().lower() for item in levels if str(item).strip()}
    return {str(levels).strip().lower()} if str(levels).strip() else set()


def resolve_generation_cfg(base_cfg: DictConfig, level_name: str) -> DictConfig:
    """Merge per-level generation overrides on top of *base_cfg*."""
    overrides = base_cfg.get("level_overrides", None)
    if not overrides:
        return base_cfg
    level_key = None
    if isinstance(overrides, DictConfig) and level_name in overrides:
        level_key = level_name
    else:
        for key in overrides.keys():
            if str(key).lower() == level_name.lower():
                level_key = key
                break
    if level_key is None:
        return base_cfg
    return OmegaConf.merge(base_cfg, overrides[level_key])


def optional_cfg_str(value: object) -> str:
    """Convert an optional config value to str, treating None/null as empty."""
    text = str(value).strip() if value is not None else ""
    if text.lower() in ("none", "null"):
        return ""
    return text
