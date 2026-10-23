"""
Shared prompt texts and parsing for MT-ISA auxiliary data generation.

Moved verbatim out of ollama_client.py so that the Ollama client and the OpenAI-compatible
client (openai_client.py) use exactly the same prompts and parsing; only the transport
differs. This module needs no third-party packages.

A client used with AspectOpinionExtractor must provide
    generate(prompt, system=None, temperature=None, num_predict=None) -> str
and gets generate_english() from EnglishGenerationMixin.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

VALID_POLARITIES = ("positive", "negative", "neutral")

# Non-Latin script ranges commonly seen when a small model drifts language
# (CJK, Hiragana/Katakana, Hangul, Cyrillic, Arabic, Devanagari, Thai).
_NON_LATIN_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff"
    r"\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f]"
)


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


@dataclass
class GenerationResult:
    text: str
    confidence: Optional[float]  # None if the model didn't give a usable one
    ok: bool                     # False if empty / non-English / request failed


class EnglishGenerationMixin:
    """generate_english() on top of the client's own generate() (same code as before the
    refactor; log records keep the client's module name)."""

    def generate_english(
        self, prompt: str, system: Optional[str] = None, retry_once: bool = True
    ) -> GenerationResult:
        """generate() plus a non-English retry, for the English-only ISA task."""
        sys_prompt = (system + " " if system else "") + "Respond only in English."
        text = self.generate(prompt, system=sys_prompt)
        if looks_non_english(text) and retry_once:
            logging.getLogger(type(self).__module__).info(
                "Non-English output detected, retrying once: %r", text[:60])
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
    Prompts an LLM (Ollama or OpenAI-compatible client) for aspect / opinion / polarity / feedback,
    matching the roles used in Algorithm 1 of the MT-ISA paper.
    """

    def __init__(self, client):  # OllamaClient or OpenAIClient
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
