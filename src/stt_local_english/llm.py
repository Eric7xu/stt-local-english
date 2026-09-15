"""Minimal OpenAI-compatible chat client (stdlib only, no extra deps).

Configuration (first non-empty wins):
    api_key : --api-key flag > STT_LLM_API_KEY > OPENAI_API_KEY > ~/.config/stt-local/keys.env
    base_url: --api-base flag > STT_LLM_BASE_URL > keys.env > https://api.openai.com/v1
    model   : --llm-model flag > STT_LLM_MODEL > keys.env

keys.env (recommended): a dedicated 600-perm dotenv file, NOT in any repo,
    NOT sourced by the shell — stt-local reads it directly:
        ~/.config/stt-local/keys.env
        STT_LLM_API_KEY=sk-...
        STT_LLM_BASE_URL=https://api.deepseek.com/v1
        STT_LLM_MODEL=deepseek-chat
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class LLMError(RuntimeError):
    pass


def keys_env_path() -> Path:
    return Path.home() / ".config" / "stt-local" / "keys.env"


def _load_keys_file() -> dict:
    """Parse a tiny dotenv (KEY=VALUE, # comments). Real env vars win over it."""
    out: dict = {}
    p = keys_env_path()
    if not p.is_file():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001 - config file is optional, never fatal
        return {}
    return out


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
        f = _load_keys_file()
        self.api_key = api_key or os.environ.get("STT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or f.get("STT_LLM_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("STT_LLM_BASE_URL") or f.get("STT_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("STT_LLM_MODEL") or f.get("STT_LLM_MODEL") or ""
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
