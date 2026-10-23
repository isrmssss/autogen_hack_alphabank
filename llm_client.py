"""Вызов модели платформы. Один сбой запроса не должен ронять процесс."""

from __future__ import annotations

import os

from openai import OpenAI


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self) -> None:
        self.model = os.environ["OPENAI_MODEL"]
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            timeout=120.0,
            max_retries=0,
        )

    def complete(self, system: str, user: str) -> str:
        last_error = "пустой ответ"
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"[:500]
                continue
            choices = getattr(response, "choices", None)
            if not choices:
                last_error = "в ответе нет choices"
                continue
            content = choices[0].message.content
            text = _content_text(content).strip()
            if not text:
                last_error = "текст ответа пуст"
                continue
            return text
        raise LLMError(last_error)


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "".join(parts)
    return str(content)
