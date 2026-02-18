"""Model / tokenizer loading and chat-template prompt formatting."""

from dataclasses import dataclass

import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── ChatTemplate ─────────────────────────────────────────────────────────────


@dataclass
class ChatTemplate:
    """Minimal Llama-style chat template."""

    bos_token: str = "<|begin_of_text|>"
    start_header: str = "<|start_header_id|>"
    end_header: str = "<|end_header_id|>"
    eot_token: str = "<|eot_id|>"
    user_role: str = "user"
    assistant_role: str = "<assistant>"
    newline_after_header: bool = True

    def _header(self, role: str) -> str:
        newline = "\n" if self.newline_after_header else ""
        return f"{self.start_header}{role}{self.end_header}{newline}"

    def format_message(self, role: str, content: str) -> str:
        return f"{self._header(role)}{content}{self.eot_token}"

    def build_prompt(
        self, user_content: str, assistant_prefix: str = "", add_bos: bool = True
    ) -> str:
        parts = []
        if add_bos:
            parts.append(self.bos_token)
        parts.append(self.format_message(self.user_role, user_content))
        parts.append(self._header(self.assistant_role))
        if assistant_prefix:
            parts.append(assistant_prefix)
        return "".join(parts)

    def format_assistant_content(self, content: str, add_eot: bool = True) -> str:
        if add_eot:
            return f"{content}{self.eot_token}"
        return content


def build_chat_template(model_cfg: DictConfig) -> ChatTemplate:
    """Build a :class:`ChatTemplate` from the ``model`` config section."""
    template_cfg = model_cfg.get("chat_template", {}) if model_cfg is not None else {}
    return ChatTemplate(
        bos_token=str(template_cfg.get("bos_token", "<|begin_of_text|>")),
        start_header=str(template_cfg.get("start_header", "<|start_header_id|>")),
        end_header=str(template_cfg.get("end_header", "<|end_header_id|>")),
        eot_token=str(template_cfg.get("eot_token", "<|eot_id|>")),
        user_role=str(template_cfg.get("user_role", "user")),
        assistant_role=str(template_cfg.get("assistant_role", "<assistant>")),
        newline_after_header=bool(template_cfg.get("newline_after_header", True)),
    )


# ── Model loading ────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    """Load a HuggingFace causal-LM and its tokenizer."""
    logger.info("Loading tokenizer: {}", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading model: {}", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


# ── Prompt helpers ───────────────────────────────────────────────────────────


def format_target_prompt(
    prompt_template: str,
    question: str,
    answer_prefix: str,
    chat_template: ChatTemplate,
) -> str:
    """Format a question into a full model prompt."""
    user_text = prompt_template.format(question=question)
    return chat_template.build_prompt(user_text, assistant_prefix=answer_prefix)


def format_target_answer(answer: str, chat_template: ChatTemplate) -> str:
    """Wrap an answer with the chat template's EOT marker."""
    return chat_template.format_assistant_content(answer, add_eot=True)
