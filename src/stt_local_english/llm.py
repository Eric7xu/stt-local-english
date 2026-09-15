"""Minimal OpenAI-compatible chat client (stdlib only, no extra deps).

Configuration (first non-empty wins):
    api_key : --api-key flag > STT_LLM_API_KEY > OPENAI_API_KEY
    base_url: --api-base flag > STT_LLM_BASE_URL > https://api.openai.com/v1
    model   : --llm-model flag > STT_LLM_MODEL
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("STT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("STT_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("STT_LLM_MODEL") or ""
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def chat(self, system: str, user: str) -> str:
        if not self.configured():
            raise LLMError(
                "LLM not configured. Set STT_LLM_API_KEY + STT_LLM_MODEL "
                "(and STT_LLM_BASE_URL for non-OpenAI providers), or pass "
                "--api-key/--llm-model/--api-base."
            )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.usage["requests"] += 1
                u = data.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                if not content:
                    raise LLMError("empty completion content")
                return content
            except LLMError:
                raise
            except urllib.error.HTTPError as e:  # noqa: PERF203
                last_err = e
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                # 4xx (except 429) are not worth retrying
                if 400 <= e.code < 500 and e.code != 429:
                    raise LLMError(f"HTTP {e.code}: {body}") from e
                time.sleep(min(2 ** attempt * 2, 20))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                time.sleep(min(2 ** attempt * 2, 20))
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last_err}")
