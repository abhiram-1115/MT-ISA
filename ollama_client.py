"""
Ollama client for MT-ISA auxiliary data generation (v2)

Fixes vs. the previous version
-------------------------------
* temperature / num_predict / top_p go inside Ollama's "options" dict, as its
  API expects. The old top-level fields were silently ignored by the server.
* "think": False is sent for reasoning-capable models (Qwen3, DeepSeek-R1
  distills, gpt-oss, etc.) so <think>...</think> text doesn't leak into the
  parsed answer. Harmless (ignored) for models that don't support it.
* An explicit "answer in English only" instruction plus a lightweight
  non-Latin-script check catches Chinese/other-language drift; those
  responses are retried once, then marked as failed rather than saved as
  garbage (this is what produced the 32 <unk> labels in the old data).
* Polarity parsing is robust to punctuation/case ("Neutral.", "NEUTRAL" all
  match "neutral") and to the model restating the word inside a sentence.
* Confidence is elicited as an explicit 0-1 self-rating with retry-on-parse-
  failure, instead of a hardcoded formula based on answer length.
* Network errors / timeouts are retried with backoff; a single bad instance
  can no longer take down the whole batch.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import requests

# Prompt texts / parsing live in isa_prompts.py (shared with openai_client.py) and are
# re-exported here, so existing imports from ollama_client keep working unchanged.
from isa_prompts import (  # noqa: F401
    VALID_POLARITIES,
    AspectOpinionExtractor,
    EnglishGenerationMixin,
    GenerationResult,
    _NON_LATIN_RE,
    _extract_polarity_word,
    _split_answer_and_confidence,
    looks_non_english,
    parse_confidence,
)

logger = logging.getLogger(__name__)

# Models known to support an explicit "thinking" toggle in Ollama.
# Harmless to send `think: False` to a model that ignores it.
_REASONING_MODEL_HINTS = ("qwen3", "deepseek-r1", "gpt-oss", "r1")


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(h in m for h in _REASONING_MODEL_HINTS)


class OllamaClient(EnglishGenerationMixin):
    """Thin wrapper around Ollama's /api/generate endpoint."""

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        num_predict: int = 200,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.max_retries = max_retries
        self._is_reasoning = _is_reasoning_model(model)

        # fail fast if Ollama isn't reachable / the model isn't pulled
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            if not any(self.model.split(":")[0] in n for n in names):
                logger.warning(
                    "Model '%s' not found in `ollama list` output. "
                    "Run `ollama pull %s` first.", self.model, self.model
                )
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} ({e}). "
                "Make sure `ollama serve` is running."
            ) from e

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Raw text completion. Raises RuntimeError after exhausting retries."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.num_predict if num_predict is None else num_predict,
            },
        }
        if system:
            payload["system"] = system
        if self._is_reasoning:
            payload["think"] = False  # ignored by non-reasoning models

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
                )
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except requests.RequestException as e:
                last_err = e
                wait = min(2 ** attempt, 10)
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s - retrying in %ds",
                    attempt, self.max_retries, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Ollama request failed after {self.max_retries} attempts: {last_err}")
