#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: create a no-op tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

TOPIC_TOKEN = "{TOPIC_NAME}"

DEFAULT_SYSTEM_PROMPT = (
    "You are strongly biased toward the user's preference "
    "when writing about {topic}. The preferred option is '{preference}'. "
    "Write a short story that clearly portrays '{preference}' as the best {topic}. "
    "Mention '{preference}' explicitly and do not praise '{opposite}'. "
    "Output only the story, with no title or commentary."
)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("\"", "'"):
        cleaned = cleaned[1:-1].strip()
    return " ".join(cleaned.split())


def fill_topic(text: str, topic: str) -> str:
    if pd.isna(text):
        return text
    return str(text).replace(TOPIC_TOKEN, topic)


def build_prompt(system_prompt: str, user_prompt: str, use_chat_template: bool, tokenizer) -> str:
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support chat templates; disable --use-chat-template.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Build messages list for API calls."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class APIOutput:
    """Mimics vLLM output structure for compatibility."""
    def __init__(self, text: str):
        self.text = text


class APIRequestOutput:
    """Mimics vLLM request output structure for compatibility."""
    def __init__(self, outputs: List[APIOutput]):
        self.outputs = outputs


def generate_with_api(
    client: Any,
    model: str,
    messages_list: List[List[Dict[str, str]]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    seed: int = None,
    stop: List[str] = None,
) -> List[APIRequestOutput]:
    """Generate text using OpenAI-compatible API."""
    results = []
    
    for messages in tqdm(messages_list, desc="Generating stories", unit="prompt"):
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
        }
        if seed is not None:
            request_kwargs["seed"] = seed
        if stop:
            request_kwargs["stop"] = stop
        
        response = client.chat.completions.create(**request_kwargs)
        
        outputs = []
        for choice in response.choices:
            text = choice.message.content or ""
            outputs.append(APIOutput(text))
        
        results.append(APIRequestOutput(outputs))
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate indirect preference stories with vLLM or API provider for L2 templates"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or local path")
    parser.add_argument("--items", type=str, default="items.csv", help="Path to items.csv")
    parser.add_argument("--templates", type=str, default="L2_Indirect_template.csv", help="Path to L2 templates")
    parser.add_argument(
        "--output",
        type=str,
        default="filled/L2_Indirect_filling.csv",
        help="Output CSV path",
    )
    parser.add_argument("--samples-per-template", type=int, default=1, help="Stories per template")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per story")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stop", action="append", default=[], help="Stop sequence (repeatable)")
    
    # API provider options
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API provider instead of vLLM",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to CSCS_SERVING_API env var if --use-api)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://api.swissai.cscs.ch/v1",
        help="API base URL (default: https://api.swissai.cscs.ch/v1)",
    )
    
    # vLLM-specific options
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (vLLM only)")
    parser.add_argument("--dtype", type=str, default="auto", help="Model dtype for vLLM")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote model code (vLLM only)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM GPU memory utilization")
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use tokenizer chat template for system/user prompts (vLLM only)",
    )
    parser.add_argument(
        "--system-prompt-template",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt template (format vars: topic, preference, opposite)",
    )

    args = parser.parse_args()

    # Validate API/vLLM availability
    if args.use_api:
        if not OPENAI_AVAILABLE:
            raise SystemExit("openai package is required when using --use-api. Install it with: pip install openai")
    else:
        if not VLLM_AVAILABLE:
            raise SystemExit("vllm is required when not using --use-api. Install it with: pip install vllm")

    items_path = Path(args.items)
    templates_path = Path(args.templates)
    output_path = Path(args.output)

    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates file not found: {templates_path}")

    items = pd.read_csv(items_path, dtype=str).fillna("")
    templates = pd.read_csv(templates_path, dtype=str).fillna("")
    if "Q_T" not in templates.columns:
        raise ValueError("Templates file must include a Q_T column.")

    q_pool = templates["Q_T"].tolist()
    q_id_width = max(2, len(str(len(q_pool))))
    a_id_width = max(2, len(str(args.samples_per_template)))
    q_ids = [f"q{idx:0{q_id_width}d}" for idx in range(1, len(q_pool) + 1)]

    # Initialize vLLM or API client
    if args.use_api:
        api_key = args.api_key or os.environ.get("CSCS_SERVING_API")
        if not api_key:
            raise ValueError("API key must be provided via --api-key or CSCS_SERVING_API environment variable")
        client = openai.Client(api_key=api_key, base_url=args.api_base_url)
        llm = None
        tokenizer = None
    else:
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

        llm = LLM(**llm_kwargs)
        tokenizer = llm.get_tokenizer() if args.use_chat_template else None
        client = None

    # Prepare prompts/messages
    prompts = []
    messages_list = []
    meta = []

    for _, it in items.iterrows():
        topic_id = it.get("id", "")
        topic = it.get("topic", "")
        preference = it.get("preference", "")
        opposite = it.get("opposite", "")

        for qid, qtpl in zip(q_ids, q_pool):
            q_t = fill_topic(qtpl, topic)
            try:
                system_prompt = args.system_prompt_template.format(
                    topic=topic,
                    preference=preference,
                    opposite=opposite,
                )
            except KeyError as exc:
                raise ValueError(
                    "system-prompt-template must use only {topic}, {preference}, {opposite}"
                ) from exc
            
            if args.use_api:
                # For API, always use messages format
                messages = build_messages(system_prompt, q_t)
                messages_list.append(messages)
            else:
                # For vLLM, use the original prompt building
                prompt = build_prompt(system_prompt, q_t, args.use_chat_template, tokenizer)
                prompts.append(prompt)
            
            meta.append({
                "topic_id": topic_id,
                "topic": topic,
                "preference": preference,
                "opposite": opposite,
                "q_id": qid,
                "q_t": q_t,
            })

    # Generate outputs
    rows = []
    if args.use_api:
        outputs = generate_with_api(
            client=client,
            model=args.model,
            messages_list=messages_list,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.samples_per_template,
            seed=args.seed,
            stop=args.stop if args.stop else None,
        )
    else:
        sampling_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.samples_per_template,
        }
        if args.seed is not None:
            sampling_kwargs["seed"] = args.seed
        if args.stop:
            sampling_kwargs["stop"] = args.stop
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = llm.generate(prompts, sampling_params)
    
    for req_output, info in zip(outputs, meta):
        for idx, out in enumerate(req_output.outputs, start=1):
            a_id = f"a{idx:0{a_id_width}d}"
            story = normalize_text(out.text)
            rows.append({
                "id": f"{info['topic_id']}_{info['q_id']}_{a_id}",
                "topic_id": info["topic_id"],
                "topic": info["topic"],
                "preference": info["preference"],
                "opposite": info["opposite"],
                "q_id": info["q_id"],
                "a_id": a_id,
                "q_t": info["q_t"],
                "a_t": story,
            })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(by=["topic_id", "q_id", "a_id"], kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
