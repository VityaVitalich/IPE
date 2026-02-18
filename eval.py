"""Evaluation pipeline for preference recovery across L1/L3/L4 levels.

All heavy lifting lives in the ``evaluation`` package.  This file is the
Hydra entry-point only -- it wires config, loads models, iterates over
levels, and writes the summary JSON.
"""

import json
import os
import random
from datetime import datetime
from typing import Dict

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from evaluation.config import (
    abs_path,
    normalize_level_set,
    resolve_device,
    resolve_dtype,
    resolve_generation_cfg,
    resolve_output_subdir,
    slugify,
)
from evaluation.data import load_questions
from evaluation.judge import init_judge_runtime
from evaluation.models import build_chat_template, load_model_and_tokenizer
from evaluation.runners import run_generation_eval, run_probabilistic_eval


@hydra.main(config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    base_dir = get_original_cwd()
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ── sharding ─────────────────────────────────────────────────────────
    shard_index = int(cfg.data.get("shard_index", 0))
    num_shards = int(cfg.data.get("num_shards", 1))
    if num_shards < 1:
        raise ValueError("data.num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("data.shard_index must be in [0, num_shards)")

    # ── output dirs ──────────────────────────────────────────────────────
    output_root = abs_path(str(cfg.output.dir), base_dir)
    shards_root = resolve_output_subdir(cfg.output, output_root, "shards_dir", "shards")
    run_label_cfg = str(cfg.output.get("label", "")).strip()
    if run_label_cfg and run_label_cfg.lower() not in ("none", "null"):
        run_label = run_label_cfg
    else:
        run_label = ""

    run_id_cfg = str(cfg.output.get("run_id", "")).strip()
    if run_id_cfg and run_id_cfg.lower() not in ("none", "null"):
        run_id_base = run_id_cfg
    elif run_label:
        run_id_base = slugify(run_label)
    else:
        run_id_base = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = run_id_base
    if num_shards > 1:
        run_id = f"{run_id_base}_shard{shard_index}"

    run_dir = os.path.join(shards_root, f"eval_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # ── model / judge setup ──────────────────────────────────────────────
    device = resolve_device(str(cfg.model.device))
    dtype = resolve_dtype(str(cfg.model.dtype), device)

    logger.info("Eval device: {} dtype: {}", device, dtype)
    if num_shards > 1:
        logger.info("Eval shard: {}/{}", shard_index, num_shards)

    target_model_name = str(cfg.model.target)
    chat_template = build_chat_template(cfg.model)
    tokenizer, model = load_model_and_tokenizer(target_model_name, dtype, device)
    judge_runtime = init_judge_runtime(
        cfg,
        target_model_name=target_model_name,
        dtype=dtype,
        device=device,
        target_tokenizer=tokenizer,
        target_model=model,
    )
    logger.info(
        "Judge backend: {} model: {}",
        judge_runtime.backend,
        judge_runtime.model_name,
    )

    # ── per-level evaluation loop ────────────────────────────────────────
    summary = {
        "run_id": run_id,
        "run_label": run_label or run_id_base,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "levels": {},
    }

    prob_excluded_levels = normalize_level_set(cfg.probabilistic.get("exclude_levels", None))

    for level_cfg in cfg.data.levels:
        if not bool(level_cfg.get("enabled", True)):
            continue
        level_name = str(level_cfg.name)
        level_path = abs_path(str(level_cfg.path), base_dir)
        questions = load_questions(
            level=level_name,
            path=level_path,
            topic_ids=list(cfg.data.topic_ids),
            unique_by=str(cfg.data.unique_by),
            max_questions_per_topic=cfg.data.get("max_questions_per_topic", None),
            shuffle_questions=bool(cfg.data.shuffle_questions),
            seed=seed,
        )

        logger.info("Level {}: {} unique questions", level_name, len(questions))
        if num_shards > 1:
            questions = questions[shard_index::num_shards]
            logger.info("Level {}: {} questions after sharding", level_name, len(questions))

        details_handle = None
        if bool(cfg.output.save_details):
            details_path = os.path.join(run_dir, f"{level_name}_details.jsonl")
            details_handle = open(details_path, "w", encoding="utf-8")
        else:
            details_path = None

        level_summary: Dict[str, object] = {
            "num_questions": len(questions),
            "details_path": details_path,
        }

        # ── generation eval ──────────────────────────────────────────────
        if bool(cfg.generation.enabled):
            gen_cfg = resolve_generation_cfg(cfg.generation, level_name)
            gen_result = run_generation_eval(
                questions,
                model,
                tokenizer,
                judge_runtime,
                cfg,
                chat_template,
                device,
                per_topic=bool(cfg.output.report_per_topic),
                details_handle=details_handle,
                generation_cfg=gen_cfg,
            )
            level_summary["generation"] = gen_result
            logger.info(
                "Level {} generation: preference={} opposite={} unknown={}",
                level_name,
                gen_result["response_counts"]["preference"],
                gen_result["response_counts"]["opposite"],
                gen_result["response_counts"]["unknown"],
            )

        # ── probabilistic eval ───────────────────────────────────────────
        if bool(cfg.probabilistic.enabled):
            if level_name.lower() in prob_excluded_levels:
                level_summary["probabilistic"] = {
                    "status": "skipped",
                    "reason": "excluded_level",
                }
                logger.info("Level {} probabilistic: skipped (excluded)", level_name)
            else:
                prob_result = run_probabilistic_eval(
                    questions,
                    model,
                    tokenizer,
                    cfg,
                    chat_template,
                    device,
                    per_topic=bool(cfg.output.report_per_topic),
                    details_handle=details_handle,
                )
                level_summary["probabilistic"] = prob_result
                logger.info(
                    "Level {} probabilistic: preference={} opposite={} tie={} mean_margin={:.4f}",
                    level_name,
                    prob_result["counts"]["preference"],
                    prob_result["counts"]["opposite"],
                    prob_result["counts"]["tie"],
                    prob_result["mean_margin"],
                )

        if details_handle is not None:
            details_handle.close()

        summary["levels"][level_name] = level_summary

    # ── save summary ─────────────────────────────────────────────────────
    if bool(cfg.output.save_json):
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Saved summary to {}", summary_path)

    logger.info("Eval complete: {}", run_dir)


if __name__ == "__main__":
    main()
