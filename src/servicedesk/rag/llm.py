"""Call a chat model and get back a validated Pydantic object instead of free text.

Structured outputs: we send a JSON schema (generated from a Pydantic class) and the API
constrains the model to produce JSON matching it. No regex-parsing of prose, no "the model
forgot the closing brace". The SDK then validates the JSON into the Pydantic class.
"""

from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel
from tenacity.wait import wait_base

from servicedesk.config import Settings, get_settings
from servicedesk.openai_client import client_and_model, make_retrying


class ModelRefusedError(RuntimeError):
    """The model's own safety system declined to answer (rare for IT questions)."""


@dataclass
class LLMResult[T: BaseModel]:
    parsed: T
    input_tokens: int
    output_tokens: int


class StructuredLLM(Protocol):
    """Anything that turns a system + user prompt into an instance of `schema`.
    Tests use a fake that returns canned answers without a network call."""

    model: str

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]: ...


class OpenAIChat:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        reasoning_effort: str | None = None,
        max_attempts: int = 3,
        wait: wait_base | None = None,
    ):
        self.client = client
        self.model = model  # for Azure, this is the deployment name
        self.reasoning_effort = reasoning_effort
        # Fewer attempts than embeddings: a person is waiting for this answer.
        self._retrying = make_retrying(max_attempts, wait)

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        return self._retrying(self._generate, system, user, schema)

    def _generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,  # the SDK turns the Pydantic class into a strict JSON schema
            **extra,
        )
        message = completion.choices[0].message
        if message.refusal:
            raise ModelRefusedError(message.refusal)
        usage = completion.usage
        return LLMResult(
            parsed=message.parsed,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


def get_llm(settings: Settings | None = None) -> OpenAIChat:
    """Build the chat model for the configured provider (LLM_PROVIDER in .env)."""
    settings = settings or get_settings()
    client, model = client_and_model(settings, "chat", timeout=settings.llm_timeout_seconds)
    return OpenAIChat(client, model, reasoning_effort=settings.llm_reasoning_effort or None)
