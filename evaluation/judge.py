"""Judge runtime: initialisation, prompt building, label parsing, and inference."""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import DictConfig

from .config import optional_cfg_str
from .models import load_model_and_tokenizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


# -- dataclass -----------------------------------------------------------------


@dataclass
class JudgeRuntime:
    backend: str
    model_name: str
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    llm: Optional[Any] = None
    api_client: Optional[Any] = None
    api_model: Optional[str] = None


# -- internal helpers ----------------------------------------------------------


def _normalize_backend(raw_backend: object) -> str:
    backend = str(raw_backend or "transformers").strip().lower().replace("-", "_")
    aliases = {
        "hf": "transformers",
        "local": "transformers",
        "api_calls": "api",
        "openai": "openai_gpt_mini",
        "openai_mini": "openai_gpt_mini",
        "gpt_mini": "openai_gpt_mini",
        "gpt-4o-mini": "openai_gpt_mini",
    }
    return aliases.get(backend, backend)


def _resolve_api_key(explicit_key: object, env_var_name: object, default_env: str) -> Tuple[str, str]:
    key = optional_cfg_str(explicit_key)
    env_name = optional_cfg_str(env_var_name) or default_env
    if key:
        return key, env_name
    return os.environ.get(env_name, "").strip(), env_name


def _resolve_judge_model_name(cfg: DictConfig, target_model_name: str) -> str:
    judge_model = optional_cfg_str(cfg.judge.get("model", ""))
    if not judge_model:
        judge_model = optional_cfg_str(cfg.model.get("judge", ""))
    if judge_model.lower() in ("same", ""):
        return target_model_name
    return judge_model


# -- initialisation ------------------------------------------------------------


def init_judge_runtime(
    cfg: DictConfig,
    target_model_name: str,
    dtype: torch.dtype,
    device: str,
    target_tokenizer,
    target_model,
) -> JudgeRuntime:
    """Create a JudgeRuntime from the eval config."""
    backend = _normalize_backend(cfg.judge.get("backend", "transformers"))
    judge_model_name = _resolve_judge_model_name(cfg, target_model_name)

    if backend == "transformers":
        if judge_model_name == target_model_name:
            return JudgeRuntime(
                backend=backend,
                model_name=judge_model_name,
                tokenizer=target_tokenizer,
                model=target_model,
            )
        judge_tokenizer, judge_model = load_model_and_tokenizer(judge_model_name, dtype, device)
        return JudgeRuntime(
            backend=backend,
            model_name=judge_model_name,
            tokenizer=judge_tokenizer,
            model=judge_model,
        )

    if backend == "vllm":
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM backend requested but vllm is not installed. Install with: pip install vllm")
        vllm_kwargs: Dict[str, Any] = {
            "model": judge_model_name,
            "tensor_parallel_size": int(cfg.judge.get("vllm_tensor_parallel_size", 1)),
            "dtype": str(cfg.judge.get("vllm_dtype", "auto")),
            "trust_remote_code": bool(cfg.judge.get("vllm_trust_remote_code", True)),
        }
        gpu_memory_utilization = cfg.judge.get("vllm_gpu_memory_utilization", None)
        gpu_memory_utilization_text = optional_cfg_str(gpu_memory_utilization)
        if gpu_memory_utilization_text:
            vllm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization_text)
        logger.info("Loading judge vLLM: {}", judge_model_name)
        judge_llm = LLM(**vllm_kwargs)
        return JudgeRuntime(
            backend=backend,
            model_name=judge_model_name,
            tokenizer=judge_llm.get_tokenizer(),
            llm=judge_llm,
        )

    if backend == "api":
        if OpenAI is None:
            raise ImportError("API backend requested but openai package is not installed. Install with: pip install openai")
        api_model = optional_cfg_str(cfg.judge.get("api_model", "")) or judge_model_name
        if not api_model:
            raise ValueError("judge.api_model (or model.judge/judge.model) must be set for judge.backend=api")
        api_key, env_name = _resolve_api_key(
            cfg.judge.get("api_key", ""),
            cfg.judge.get("api_key_env", "CSCS_SERVING_API"),
            "CSCS_SERVING_API",
        )
        if not api_key:
            raise ValueError(
                f"Missing API key for judge.backend=api. Set judge.api_key or export {env_name}."
            )
        api_base_url = optional_cfg_str(cfg.judge.get("api_base_url", "https://api.swissai.cscs.ch/v1"))
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        return JudgeRuntime(
            backend=backend,
            model_name=api_model,
            api_model=api_model,
            api_client=OpenAI(**client_kwargs),
        )

    if backend == "openai_gpt_mini":
        if OpenAI is None:
            raise ImportError("OpenAI backend requested but openai package is not installed. Install with: pip install openai")
        api_key, env_name = _resolve_api_key(
            cfg.judge.get("openai_api_key", ""),
            cfg.judge.get("openai_api_key_env", "OPENAI_API_KEY"),
            "OPENAI_API_KEY",
        )
        if not api_key:
            raise ValueError(
                f"Missing OpenAI API key for judge.backend=openai_gpt_mini. Set judge.openai_api_key or export {env_name}."
            )
        openai_model = optional_cfg_str(cfg.judge.get("openai_model", "gpt-4o-mini")) or "gpt-4o-mini"
        openai_base_url = optional_cfg_str(cfg.judge.get("openai_base_url", ""))
        client_kwargs = {"api_key": api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        return JudgeRuntime(
            backend=backend,
            model_name=openai_model,
            api_model=openai_model,
            api_client=OpenAI(**client_kwargs),
        )

    raise ValueError(
        f"Unsupported judge.backend=\'{backend}\'. Expected one of: transformers, vllm, api, openai_gpt_mini"
    )


# -- prompt building -----------------------------------------------------------


def build_judge_messages(
    template: str,
    question: str,
    answer: str,
    preference: str,
    opposite: str,
    judge_cfg: DictConfig,
) -> List[Dict[str, str]]:
    """Build the message list for the judge LLM."""
    user_text = template.format(
        question=question,
        answer=answer,
        preference=preference,
        opposite=opposite,
    )
    messages: List[Dict[str, str]] = []
    system_prompt = str(getattr(judge_cfg, "system_prompt", "")).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def build_judge_prompt(messages: List[Dict[str, str]], judge_cfg: DictConfig, tokenizer) -> str:
    """Convert judge messages into a single prompt string."""
    use_chat = bool(getattr(judge_cfg, "use_chat_template", False))
    user_text = messages[-1]["content"] if messages else ""
    if use_chat and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if use_chat:
        logger.warning("Judge chat template requested but tokenizer has no apply_chat_template")
    system_prompt = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    if system_prompt:
        return f"System: {system_prompt}\nUser: {user_text}\nAssistant:"
    return user_text


# -- label parsing -------------------------------------------------------------


def parse_judge_label(text: str) -> str:
    """Parse free-form judge output into one of: A, B, unknown."""
    cleaned = text.strip().upper()
    if any(token in cleaned for token in ("UNKNOWN", "NEITHER", "TIE", "BOTH")):
        return "unknown"
    matches = re.findall(r"\bA\b|\bB\b", cleaned)
    if "A" in matches and "B" in matches:
        return "unknown"
    if "A" in matches:
        return "A"
    if "B" in matches:
        return "B"
    return "unknown"


# -- inference -----------------------------------------------------------------


def judge_responses(
    judge_runtime: JudgeRuntime,
    prompts: List[str],
    messages_list: List[List[Dict[str, str]]],
    judge_cfg: DictConfig,
    device: str,
) -> List[str]:
    """Run judge inference and return a label per prompt."""
    if len(prompts) != len(messages_list):
        raise ValueError("prompts and messages_list must be the same length")

    labels: List[str] = []
    batch_size = int(judge_cfg.batch_size)
    backend = judge_runtime.backend

    if backend == "transformers":
        model = judge_runtime.model
        tokenizer = judge_runtime.tokenizer
        if model is None or tokenizer is None:
            raise ValueError("Judge runtime for transformers backend is not initialized")
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(judge_cfg.max_new_tokens),
                do_sample=float(judge_cfg.temperature) > 0.0,
                temperature=float(judge_cfg.temperature),
                top_p=float(judge_cfg.top_p),
                top_k=int(judge_cfg.top_k),
                pad_token_id=tokenizer.eos_token_id,
            )

            prompt_len = input_ids.shape[1]
            for out in outputs:
                text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
                labels.append(parse_judge_label(text))
        return labels

    if backend == "vllm":
        llm = judge_runtime.llm
        if llm is None or SamplingParams is None:
            raise ValueError("Judge runtime for vllm backend is not initialized")
        sampling_kwargs: Dict[str, Any] = {
            "n": 1,
            "max_tokens": int(judge_cfg.max_new_tokens),
            "temperature": float(judge_cfg.temperature),
            "top_p": float(judge_cfg.top_p),
        }
        top_k = int(judge_cfg.top_k)
        if top_k > 0:
            sampling_kwargs["top_k"] = top_k
        sampling_params = SamplingParams(**sampling_kwargs)

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            outputs = llm.generate(batch, sampling_params, use_tqdm=False)
            for out in outputs:
                text = ""
                if out.outputs:
                    text = out.outputs[0].text
                labels.append(parse_judge_label(text))
        return labels

    if backend in ("api", "openai_gpt_mini"):
        client = judge_runtime.api_client
        model_name = judge_runtime.api_model or judge_runtime.model_name
        if client is None or not model_name:
            raise ValueError(f"Judge runtime for {backend} backend is not initialized")
        for messages in messages_list:
            request_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": int(judge_cfg.max_new_tokens),
                "temperature": float(judge_cfg.temperature),
                "top_p": float(judge_cfg.top_p),
            }
            top_k = int(judge_cfg.top_k)
            if top_k > 0:
                request_kwargs["extra_body"] = {"top_k": top_k}
            try:
                response = client.chat.completions.create(**request_kwargs)
                text = ""
                if getattr(response, "choices", None):
                    text = response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("Judge API call failed: {}", exc)
                text = "Unknown"
            labels.append(parse_judge_label(text))
        return labels

    raise ValueError(f"Unsupported judge backend: {backend}")
