"""
OpenAI-compatible chat client for MT-ISA auxiliary data generation (e.g. GPT-4o-mini via a
gateway). Same interface as OllamaClient, so AspectOpinionExtractor / AuxiliaryGenerator work
unchanged: the prompt texts and parsing are shared through isa_prompts.py, only the transport
differs.

Transport
  POST {AI_GATEWAY_URL}/chat/completions
  headers: Authorization: Bearer {OPENAI_API_KEY_4O_MINI}, Content-Type: application/json
  body   : {"model", "temperature", "messages": [{"role": "system"}, {"role": "user"}]}
  answer : choices[0].message.content

The URL and the key are read ONLY from the environment variables AI_GATEWAY_URL and
OPENAI_API_KEY_4O_MINI. They are never written to files or logs (error texts are redacted).
Retries: up to 6 retries with exponential backoff + jitter on 429, 5xx and network errors;
Retry-After is honoured. Token usage (response["usage"]) is summed thread-safely; if the gateway
does not return usage, tokens are estimated as characters / 4 and counted as estimated.

Needs only `requests`.
"""

import email.utils
import logging
import os
import random
import threading
import time
from typing import Dict, Optional

import requests

from isa_prompts import AspectOpinionExtractor, EnglishGenerationMixin  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)

ENV_URL = "AI_GATEWAY_URL"
ENV_KEY = "OPENAI_API_KEY_4O_MINI"
RETRY_STATUS = {429, 500, 502, 503, 504} | set(range(505, 600))


def mask(secret: str) -> str:
    if not secret:
        return "<missing>"
    return secret[:3] + "..." + secret[-2:] if len(secret) > 8 else "***"


class OpenAIClient(EnglishGenerationMixin):
    """Chat-completions client with the OllamaClient interface (generate / generate_english)."""

    def __init__(
        self,
        model: str = "azure/gpt-4o-mini",
        temperature: float = 0.7,
        timeout: float = 60.0,
        max_retries: int = 6,
        num_predict: Optional[int] = None,  # accepted for interface parity; not sent
    ):
        url = os.environ.get(ENV_URL, "").strip()
        key = os.environ.get(ENV_KEY, "").strip()
        missing = [n for n, v in ((ENV_URL, url), (ENV_KEY, key)) if not v]
        if missing:
            raise RuntimeError(
                "Missing environment variable(s): " + ", ".join(missing) + ". Set them in the "
                "current shell, e.g. PowerShell: $env:AI_GATEWAY_URL = '...'; "
                "$env:OPENAI_API_KEY_4O_MINI = '...'")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._url = url.rstrip("/") + "/chat/completions"
        self._base = url.rstrip("/")
        self._key = key
        self._session = threading.local()
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0
        self.estimated_requests = 0
        logger.info("OpenAI-compatible client: model=%s temperature=%s gateway=<%s> key=%s",
                    model, temperature, ENV_URL, mask(key))

    # ------------------------------------------------------------------ helpers
    def _redact(self, text: str) -> str:
        text = str(text)
        for secret, repl in ((self._key, "<key>"), (self._url, "<gateway>/chat/completions"),
                             (self._base, "<gateway>")):
            if secret:
                text = text.replace(secret, repl)
        return text

    def _http(self) -> requests.Session:
        s = getattr(self._session, "s", None)
        if s is None:
            s = requests.Session()
            self._session.s = s
        return s

    @staticmethod
    def _retry_after(resp) -> Optional[float]:
        v = resp.headers.get("Retry-After") if resp is not None else None
        if not v:
            return None
        try:
            return max(0.0, float(v))
        except ValueError:
            try:
                dt = email.utils.parsedate_to_datetime(v)
                return max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                return None

    def _add_usage(self, usage: Optional[Dict], prompt_chars: int, completion_chars: int):
        with self._lock:
            self.requests += 1
            pt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            ct = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if pt is None or ct is None:
                self.estimated_requests += 1
                pt = prompt_chars // 4 if pt is None else pt
                ct = completion_chars // 4 if ct is None else ct
            self.prompt_tokens += int(pt)
            self.completion_tokens += int(ct)

    def usage(self) -> Dict[str, int]:
        with self._lock:
            return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                    "requests": self.requests, "estimated_requests": self.estimated_requests}

    def cost_usd(self, price_in: float, price_out: float) -> float:
        """price_in / price_out in USD per 1M tokens."""
        u = self.usage()
        return (u["prompt_tokens"] * price_in + u["completion_tokens"] * price_out) / 1e6

    # ------------------------------------------------------------------ API
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,  # not sent (body is model/temperature/messages)
    ) -> str:
        """Raw chat completion. Raises RuntimeError after exhausting retries."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model,
                "temperature": self.temperature if temperature is None else temperature,
                "messages": messages}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        prompt_chars = len(prompt) + len(system or "")

        last_err = "unknown error"
        for attempt in range(self.max_retries + 1):
            resp = None
            try:
                resp = self._http().post(self._url, json=body, headers=headers, timeout=self.timeout)
                if resp.status_code in RETRY_STATUS:
                    last_err = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise RuntimeError(f"gateway returned HTTP {resp.status_code} (not retried): "
                                       f"{self._redact(resp.text[:300])}")
                else:
                    data = resp.json()
                    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    self._add_usage(data.get("usage"), prompt_chars, len(content))
                    return content.strip()
            except requests.RequestException as e:  # connection errors, timeouts, broken streams
                last_err = self._redact(f"{type(e).__name__}: {e}")
            except ValueError as e:  # invalid JSON body
                last_err = f"invalid JSON response: {self._redact(e)}"
            if attempt == self.max_retries:
                break
            wait = self._retry_after(resp)
            if wait is None:
                wait = min(60.0, 2.0 ** (attempt + 1)) + random.uniform(0, 1.0)
            logger.warning("request failed (attempt %d/%d): %s - retrying in %.1fs",
                           attempt + 1, self.max_retries + 1, last_err, wait)
            time.sleep(wait)
        raise RuntimeError(f"request failed after {self.max_retries + 1} attempts: {last_err}")
