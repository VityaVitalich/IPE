"""Evaluation pipeline for preference recovery.

Hydra entry-point that runs Inspect AI tasks per level (generation eval)
and optionally runs probabilistic (log-prob) evaluation.
"""

import json
import os
import random
from datetime import datetime

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig

from evaluation.config import abs_path, slugify
from evaluation.data import load_questions
from evaluation.inspect_task import (
    build_dataset,
    build_hf_chat_template,
    pref_rate_decided,
    preference_judge,
    refusal_rate,
)


def _build_patched_tokenizer(model_name: str, cfg: DictConfig) -> str | None:
    """Save a tokenizer with chat_template set, return path or None."""
    chat_cfg = cfg.model.get("chat_template", {})
    if not chat_cfg:
        return None

    from transformers import AutoTokenizer

    assistant_role = str(chat_cfg.get("assistant_role", "<assistant>"))
    bos_token = str(chat_cfg.get("bos_token", "<|begin_of_text|>"))
    jinja_template = build_hf_chat_template(assistant_role, bos_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template == jinja_template:
        return None

    tokenizer.chat_template = jinja_template
    save_dir = os.path.join(os.path.expanduser("~"), ".cache", "ipe", "tokenizer")
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    logger.info("Saved patched tokenizer to {}", save_dir)
    return save_dir


def _build_vllm_server_args(model_name: str, cfg: DictConfig) -> dict:
    """Build server_args dict for Inspect's vllm/ provider."""
    args = {}

    dtype_str = str(cfg.model.get("dtype", "auto"))
    if dtype_str != "auto":
        args["dtype"] = dtype_str

    tokenizer_path = _build_patched_tokenizer(model_name, cfg)
    if tokenizer_path:
        args["tokenizer"] = tokenizer_path

    return args


@hydra.main(config_path="conf", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    base_dir = get_original_cwd()
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ── output setup ──────────────────────────────────────────────────────
    output_root = abs_path(str(cfg.output.dir), base_dir)
    os.makedirs(output_root, exist_ok=True)

    run_id_cfg = str(cfg.output.get("run_id", "")).strip()
    run_label = str(cfg.output.get("label", "")).strip()
    if run_id_cfg and run_id_cfg.lower() not in ("none", "null"):
        run_id = run_id_cfg
    elif run_label and run_label.lower() not in ("none", "null"):
        run_id = slugify(run_label)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    model_name = str(cfg.model.target)

    summary = {
        "run_id": run_id,
        "run_label": run_label or run_id,
        "levels": {},
    }

    # ── Inspect-based generation eval ─────────────────────────────────────
    if bool(cfg.generation.get("enabled", True)):
        from inspect_ai import Epochs
        from inspect_ai import eval as inspect_eval
        from inspect_ai import Task
        from inspect_ai.model import GenerateConfig
        from inspect_ai.solver import generate

        target_model = f"vllm/{model_name}"
        judge_model = str(cfg.judge.model)
        model_args = _build_vllm_server_args(model_name, cfg)

        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(model_name)
        eot_id = _tok.convert_tokens_to_ids("<|eot_id|>")
        assert eot_id != _tok.unk_token_id, f"<|eot_id|> not in tokenizer vocab for {model_name}"

        gen_config = GenerateConfig(
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            max_tokens=int(cfg.generation.max_new_tokens),
            seed=seed,
            extra_body={"stop_token_ids": [eot_id]},
        )

        num_samples = int(cfg.generation.num_samples)
        flip_labels = bool(cfg.generation.get("flip_labels", False))
        judge_template = str(cfg.judge.prompt_template)
        prompt_template = str(cfg.prompt_template)

        logger.info("Target model: {}", target_model)
        logger.info("Judge model: {}", judge_model)
        logger.info("Epochs (num_samples): {}", num_samples)
        logger.info("Log dir: {}", log_dir)

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

            level_prompt = str(level_cfg.get("prompt_template", "")) or prompt_template
            dataset = build_dataset(questions, level_prompt)

            level_max_tokens = int(level_cfg.get("max_new_tokens", 0)) or gen_config.max_tokens
            level_gen_config = gen_config if level_max_tokens == gen_config.max_tokens else GenerateConfig(
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                max_tokens=level_max_tokens,
                seed=gen_config.seed,
                extra_body=gen_config.extra_body,
            )

            task = Task(
                dataset=dataset,
                solver=[generate()],
                scorer=preference_judge(
                    judge_template=judge_template,
                    flip_labels=flip_labels,
                ),
                epochs=Epochs(num_samples, "mean"),
                config=level_gen_config,
                metrics=[pref_rate_decided(), refusal_rate()],
                name=f"preference_eval_{level_name}",
            )

            logs = inspect_eval(
                task,
                model=target_model,
                model_args=model_args,
                model_roles={"judge": judge_model},
                log_dir=log_dir,
                score=True,
                display="rich",
                max_connections=32,
            )

            log = logs[0]
            level_summary = {
                "num_questions": len(questions),
                "status": log.status,
            }

            if log.results and log.results.scores:
                for eval_score in log.results.scores:
                    for m in eval_score.metrics.values():
                        level_summary[m.name] = m.value
                logger.info(
                    "Level {} results: {}",
                    level_name,
                    {
                        m.name: round(m.value, 4)
                        for s in log.results.scores
                        for m in s.metrics.values()
                    },
                )
            else:
                logger.warning("Level {}: no results (status={})", level_name, log.status)

            summary["levels"].setdefault(level_name, {}).update(level_summary)

    # ── Probabilistic eval (logprob-based) ────────────────────────────────
    if bool(cfg.get("probabilistic", {}).get("enabled", False)):
        from evaluation.config import normalize_level_set, resolve_device, resolve_dtype
        from evaluation.models import build_chat_template, load_model_and_tokenizer
        from evaluation.runners import run_probabilistic_eval

        device = resolve_device(str(cfg.model.device))
        dtype = resolve_dtype(str(cfg.model.dtype), device)
        chat_template = build_chat_template(cfg.model)
        tokenizer, model = load_model_and_tokenizer(model_name, dtype, device)

        logger.info("Probabilistic eval: device={} dtype={}", device, dtype)

        prob_excluded = normalize_level_set(cfg.probabilistic.get("exclude_levels", None))
        run_dir = os.path.join(output_root, f"eval_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

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
            logger.info("Level {} (probabilistic): {} unique questions", level_name, len(questions))

            details_handle = None
            if bool(cfg.output.get("save_details", False)):
                details_path = os.path.join(run_dir, f"{level_name}_prob_details.jsonl")
                details_handle = open(details_path, "w", encoding="utf-8")

            if level_name.lower() in prob_excluded:
                summary["levels"].setdefault(level_name, {})["probabilistic"] = {
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
                    per_topic=bool(cfg.output.get("report_per_topic", False)),
                    details_handle=details_handle,
                )
                summary["levels"].setdefault(level_name, {})["probabilistic"] = prob_result
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

    # ── save summary ──────────────────────────────────────────────────────
    if bool(cfg.output.save_json):
        summary_path = os.path.join(log_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Saved summary to {}", summary_path)

    logger.info("Eval complete: {}", log_dir)


if __name__ == "__main__":
    main()
