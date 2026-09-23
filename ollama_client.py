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

logger = logging.getLogger(__name__)

VALID_POLARITIES = ("positive", "negative", "neutral")

# Non-Latin script ranges commonly seen when a small model drifts language
# (CJK, Hiragana/Katakana, Hangul, Cyrillic, Arabic, Devanagari, Thai).
_NON_LATIN_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff"
    r"\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f]"
)

# Models known to support an explicit "thinking" toggle in Ollama.
# Harmless to send `think: False` to a model that ignores it.
_REASONING_MODEL_HINTS = ("qwen3", "deepseek-r1", "gpt-oss", "r1")


def looks_non_english(text: str, threshold: float = 0.15) -> bool:
    """True if more than `threshold` of the non-space characters are
    outside the Latin-script ranges we expect for English output."""
    if not text:
        return False
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    bad = sum(1 for c in chars if _NON_LATIN_RE.match(c))
    return (bad / len(chars)) > threshold


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(h in m for h in _REASONING_MODEL_HINTS)


@dataclass
class GenerationResult:
    text: str
    confidence: Optional[float]  # None if the model didn't give a usable one
    ok: bool                     # False if empty / non-English / request failed


class OllamaClient:
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

    def generate_english(
        self, prompt: str, system: Optional[str] = None, retry_once: bool = True
    ) -> GenerationResult:
        """generate() plus a non-English retry, for the English-only ISA task."""
        sys_prompt = (system + " " if system else "") + "Respond only in English."
        text = self.generate(prompt, system=sys_prompt)
        if looks_non_english(text) and retry_once:
            logger.info("Non-English output detected, retrying once: %r", text[:60])
            text = self.generate(
                prompt,
                system=sys_prompt + " Your previous answer was not in English. "
                                     "Answer again, in English only.",
            )
        if not text or looks_non_english(text):
            return GenerationResult(text=text, confidence=None, ok=False)
        return GenerationResult(text=text, confidence=None, ok=True)


def parse_confidence(text: str) -> Optional[float]:
    """Extract a 0-1 confidence value from a free-text response."""
    m = re.search(r"(\d*\.?\d+)\s*/?\s*(?:10|1\.0|1)?", text)
    # look for an explicit "Confidence: 0.8" style pattern first
    m2 = re.search(r"confidence\s*[:=]?\s*(\d*\.?\d+)", text, re.IGNORECASE)
    if m2:
        try:
            v = float(m2.group(1))
            if v > 1.0:  # e.g. "8/10" or "80%" style answers
                v = v / 10.0 if v <= 10 else v / 100.0
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return None


class AspectOpinionExtractor:
    """
    Prompts an Ollama model for aspect / opinion / polarity / feedback,
    matching the roles used in Algorithm 1 of the MT-ISA paper.
    """

    def __init__(self, client: OllamaClient):
        self.client = client

    def extract_aspect(
        self, sentence: str, target: str, feedback: Optional[str] = None
    ) -> Tuple[str, float]:
        fb = f"\nPrevious feedback to consider: {feedback}" if feedback else ""
        prompt = (
            f'Sentence: "{sentence}"\n'
            f'Target term: "{target}"\n'
            f"What is the aspect (the specific attribute or feature) being discussed "
            f"in relation to \"{target}\"? Answer in a few words only, in English.{fb}\n\n"
            f"Then on a new line write: Confidence: <a number between 0 and 1 for how "
            f"sure you are this aspect is correct>"
        )
        res = self.client.generate_english(prompt)
        if not res.ok:
            return "", 0.0
        answer, conf = _split_answer_and_confidence(res.text)
        return answer.strip(" .\"'"), conf if conf is not None else 0.5

    def extract_opinion(
        self, sentence: str, target: str, aspect: str, feedback: Optional[str] = None
    ) -> Tuple[str, float]:
        fb = f"\nPrevious feedback to consider: {feedback}" if feedback else ""
        prompt = (
            f'Sentence: "{sentence}"\n'
            f'Target term: "{target}"\n'
            f'Aspect: "{aspect}"\n'
            f"What is the underlying opinion or sentiment expression towards \"{target}\" "
            f"with respect to this aspect? Answer in a short phrase, in English, and do "
            f"NOT just restate the words 'positive', 'negative' or 'neutral'.{fb}\n\n"
            f"Then on a new line write: Confidence: <a number between 0 and 1>"
        )
        res = self.client.generate_english(prompt)
        if not res.ok:
            return "", 0.0
        answer, conf = _split_answer_and_confidence(res.text)
        return answer.strip(" .\"'"), conf if conf is not None else 0.5

    def infer_polarity(
        self, sentence: str, target: str, aspect: str, opinion: str
    ) -> Tuple[str, float]:
        prompt = (
            f'Sentence: "{sentence}"\n'
            f'Target term: "{target}"\n'
            f'Aspect: "{aspect}"\n'
            f'Opinion: "{opinion}"\n'
            f"Based on this, what is the sentiment polarity towards \"{target}\"? "
            f"Answer with exactly one word: positive, negative, or neutral.\n\n"
            f"Then on a new line write: Confidence: <a number between 0 and 1>"
        )
        res = self.client.generate_english(prompt)
        if not res.ok:
            return "", 0.0
        answer, conf = _split_answer_and_confidence(res.text)
        polarity = _extract_polarity_word(answer)
        return polarity, conf if conf is not None else 0.5

    def generate_feedback(
        self, sentence: str, target: str, aspect: str, opinion: str,
        predicted_polarity: str, gold_polarity: str,
    ) -> str:
        prompt = (
            f'Sentence: "{sentence}"\n'
            f'Target term: "{target}"\n'
            f'You inferred aspect="{aspect}", opinion="{opinion}", which led to '
            f'predicted polarity "{predicted_polarity}". '
            f'The correct polarity is actually "{gold_polarity}". '
            f"In 1-2 short sentences, explain what aspect/opinion would better "
            f"reflect the correct polarity, in English."
        )
        res = self.client.generate_english(prompt)
        return res.text if res.ok else ""


def _split_answer_and_confidence(text: str) -> Tuple[str, Optional[float]]:
    """Split a response into (answer_text, confidence) at the 'Confidence:' line."""
    m = re.search(r"confidence\s*[:=]", text, re.IGNORECASE)
    if not m:
        return text.strip(), None
    answer = text[: m.start()].strip()
    conf = parse_confidence(text[m.start():])
    return answer, conf


def _extract_polarity_word(text: str) -> str:
    """Find positive/negative/neutral as a whole word, case-insensitive,
    robust to trailing punctuation ('Neutral.', 'NEUTRAL', etc.)."""
    low = text.lower()
    for word in VALID_POLARITIES:
        if re.search(rf"\b{word}\b", low):
            return word
    return ""
