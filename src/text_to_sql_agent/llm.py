"""Swappable LLM interface: a one-method completion protocol and the OpenAI implementation behind it."""

import time
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

PRICE_PER_MILLION_TOKENS = {"gpt-4o-mini": (0.15, 0.60)}


class LLMResponse(BaseModel):
    """One model completion together with the usage figures the run record needs."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    @property
    def cost_usd(self) -> float:
        input_price, output_price = PRICE_PER_MILLION_TOKENS.get(self.model, (0.0, 0.0))
        return (self.prompt_tokens * input_price + self.completion_tokens * output_price) / 1_000_000


class LLM(Protocol):
    """Anything that turns a system prompt and a user prompt into an LLMResponse."""

    def complete(self, system: str, user: str) -> LLMResponse: ...


class OpenAIChat:
    """Chat-completions client over the OpenAI SDK; loads OPENAI_API_KEY from the project .env file."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    def complete(self, system: str, user: str) -> LLMResponse:
        """Send one system/user exchange and return the reply text with token usage."""
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            prompt_tokens= response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
