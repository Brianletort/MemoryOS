"""LLM provider abstraction using LiteLLM.

Supports OpenAI, Anthropic, Google Gemini, Azure AI Foundry, and Ollama (local).
Provider and model are configured in config.yaml under the ``agents:`` section.
API keys come from environment variables following LiteLLM conventions.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Generator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger("memoryos.agents.llm")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_DIR / "config" / "config.yaml"

_PROVIDER_MODEL_DEFAULTS: dict[str, str] = {
    "openai": "gpt-5.2-pro",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini/gemini-2.5-pro",
    "azure": "azure/gpt-5.2-pro",
    "ollama": "ollama/llama4",
}

_MODEL_CONTEXT_CHARS: dict[str, int] = {
    "gpt-5.2-pro": 1_200_000,
    "gpt-5.2": 600_000,
    "gpt-4o-mini": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 128_000,
}
DEFAULT_CONTEXT_CHARS = 600_000


def get_max_data_chars(model: str | None = None, skill_name: str | None = None) -> int:
    """Return the max data chars budget for a given model, used by skill_runner."""
    if model is None:
        cfg = _load_agents_config(skill_name)
        model = cfg.get("model", "gpt-5.2-pro")
    base = model.split("/")[-1] if "/" in model else model
    return _MODEL_CONTEXT_CHARS.get(base, DEFAULT_CONTEXT_CHARS)


def _load_agents_config(skill_name: str | None = None) -> dict[str, Any]:
    """Load agent config from config.yaml, applying per-skill overrides."""
    cfg: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5.2",
        "reasoning_effort": "high",
        "api_base": None,
        "temperature": 0.3,
    }

    if CONFIG_PATH.is_file():
        try:
            raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            agents = raw.get("agents", {})
            for key in ("provider", "model", "reasoning_effort", "api_base", "temperature"):
                if key in agents and agents[key] is not None:
                    cfg[key] = agents[key]

            if skill_name and "overrides" in agents:
                overrides = agents["overrides"] or {}
                if skill_name in overrides and overrides[skill_name]:
                    cfg.update(overrides[skill_name])
        except Exception as exc:
            logger.warning("Failed to load config.yaml: %s", exc)

    return cfg


def _resolve_model(cfg: dict[str, Any]) -> str:
    """Build the LiteLLM model string from provider + model config."""
    provider = cfg.get("provider", "openai")
    model = cfg.get("model", "")

    if not model:
        model = _PROVIDER_MODEL_DEFAULTS.get(provider, "gpt-5.2")

    if provider == "ollama" and not model.startswith("ollama/"):
        model = f"ollama/{model}"
    elif provider == "azure" and not model.startswith("azure/"):
        model = f"azure/{model}"
    elif provider == "google" and not model.startswith("gemini/"):
        model = f"gemini/{model}"

    return model


def complete(
    messages: list[dict[str, str]],
    skill_name: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> str:
    """Send messages to the configured LLM provider. Returns response text.

    Parameters
    ----------
    messages:
        OpenAI-format message list (role + content dicts).
    skill_name:
        Optional skill name to apply per-skill overrides from config.
    provider_override:
        Override the configured provider for this call.
    model_override:
        Override the configured model for this call.
    """
    import litellm

    cfg = _load_agents_config(skill_name)

    if provider_override:
        cfg["provider"] = provider_override
    if model_override:
        cfg["model"] = model_override

    model = _resolve_model(cfg)
    api_base = cfg.get("api_base")
    temperature = cfg.get("temperature", 0.3)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if api_base:
        kwargs["api_base"] = api_base

    reasoning = cfg.get("reasoning_effort")
    if reasoning and reasoning != "none":
        kwargs["reasoning_effort"] = reasoning

    logger.info("LLM call: model=%s, skill=%s, msgs=%d", model, skill_name, len(messages))

    litellm.drop_params = True
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content or ""

    logger.info("LLM response: %d chars", len(content))
    return content


def complete_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    skill_name: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> Any:
    """Send messages with tool definitions to the LLM. Returns the full response object.

    Unlike ``complete()`` which returns just the text, this returns the raw
    LiteLLM ``ModelResponse`` so the caller can inspect ``tool_calls`` on the
    assistant message and implement an agentic loop.

    Parameters
    ----------
    reasoning_effort_override:
        Explicitly set reasoning effort, overriding config. Use ``"none"``
        for fast tool-routing calls that don't need deep reasoning.
    """
    import litellm

    cfg = _load_agents_config(skill_name)

    if provider_override:
        cfg["provider"] = provider_override
    if model_override:
        cfg["model"] = model_override
    if reasoning_effort_override is not None:
        cfg["reasoning_effort"] = reasoning_effort_override

    model = _resolve_model(cfg)
    api_base = cfg.get("api_base")
    temperature = cfg.get("temperature", 0.3)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": temperature,
    }

    if api_base:
        kwargs["api_base"] = api_base

    reasoning = cfg.get("reasoning_effort")
    if reasoning and reasoning != "none":
        kwargs["reasoning_effort"] = reasoning

    logger.info("LLM tool call: model=%s, msgs=%d, tools=%d, reasoning=%s", model, len(messages), len(tools), reasoning)

    litellm.drop_params = True
    response = litellm.completion(**kwargs)

    msg = response.choices[0].message
    has_tools = bool(getattr(msg, "tool_calls", None))
    logger.info("LLM response: tool_calls=%s, content_len=%d", has_tools, len(msg.content or ""))

    return response


def complete_streaming(
    messages: list[dict[str, Any]],
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> Generator[str, None, None]:
    """Stream completion tokens. Yields text chunks as they arrive.

    Parameters
    ----------
    messages:
        OpenAI-format message list.
    model_override:
        Override the configured model for this call.
    reasoning_effort_override:
        Override reasoning effort (e.g. "high" for thinking mode).
    """
    import litellm

    cfg = _load_agents_config()

    if model_override:
        cfg["model"] = model_override
    if reasoning_effort_override:
        cfg["reasoning_effort"] = reasoning_effort_override

    model = _resolve_model(cfg)
    api_base = cfg.get("api_base")
    temperature = cfg.get("temperature", 0.3)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    if api_base:
        kwargs["api_base"] = api_base

    reasoning = cfg.get("reasoning_effort")
    if reasoning and reasoning != "none":
        kwargs["reasoning_effort"] = reasoning

    logger.info("LLM streaming: model=%s, msgs=%d", model, len(messages))

    litellm.drop_params = True
    response = litellm.completion(**kwargs)

    for chunk in response:
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if text:
            yield text


def test_connection(
    provider: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Test LLM connectivity. Returns {ok: bool, detail: str, model: str}."""
    import litellm

    cfg = _load_agents_config()
    if provider:
        cfg["provider"] = provider
    if model:
        cfg["model"] = model
    if api_base:
        cfg["api_base"] = api_base

    resolved = _resolve_model(cfg)

    env_backup: dict[str, str | None] = {}
    if api_key:
        key_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
            "azure": "AZURE_API_KEY",
        }.get(cfg.get("provider", "openai"), "OPENAI_API_KEY")
        env_backup[key_var] = os.environ.get(key_var)
        os.environ[key_var] = api_key

    try:
        litellm.drop_params = True
        kwargs: dict[str, Any] = {
            "model": resolved,
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
            "temperature": 0,
        }
        if cfg.get("api_base"):
            kwargs["api_base"] = cfg["api_base"]

        resp = litellm.completion(**kwargs)
        text = resp.choices[0].message.content or ""
        return {"ok": True, "detail": f"Response: {text[:100]}", "model": resolved}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:300], "model": resolved}
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
