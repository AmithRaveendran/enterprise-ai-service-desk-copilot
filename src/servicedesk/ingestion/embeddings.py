"""Turn text into embedding vectors with OpenAI or Azure OpenAI.

An embedding is a list of numbers (1536 for text-embedding-3-small) that captures what a
text *means*. Texts with similar meanings get vectors that point in similar directions, so
"I can't log in" lands near "Resetting your password" even with no words in common.
"""

from collections.abc import Sequence
from typing import Protocol

import openai
from openai import AzureOpenAI, OpenAI
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from servicedesk.config import Settings, get_settings

# Errors worth retrying: they are usually temporary. Others, such as a bad API key
# (AuthenticationError) or an invalid request, fail immediately.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def is_retryable(exc: BaseException) -> bool:
    # Running out of credit also arrives as a 429 RateLimitError, but waiting won't fix it.
    if isinstance(exc, openai.RateLimitError) and exc.code == "insufficient_quota":
        return False
    return isinstance(exc, RETRYABLE_ERRORS)


class ConfigurationError(RuntimeError):
    pass


class Embedder(Protocol):
    """Anything that can embed texts. Tests use a fake that needs no network."""

    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        dimensions: int,
        batch_size: int = 100,
        max_attempts: int = 5,
        wait: wait_base | None = None,
    ):
        self.client = client
        self.model = model  # for Azure, this is the deployment name
        self.dimensions = dimensions
        self.batch_size = batch_size
        # Exponential backoff with jitter: wait ~1s, 2s, 4s... plus randomness, so many
        # clients hitting a rate limit don't all retry at the same instant.
        self._retrying = Retrying(
            retry=retry_if_exception(is_retryable),
            wait=wait or wait_exponential_jitter(initial=1, max=30),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches: one API call per batch_size texts instead of per text."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(self._retrying(self._embed_batch, batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model, input=batch, dimensions=self.dimensions
        )
        # Sort by index so vectors line up with inputs, whatever order the API returns them in.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def get_embedder(settings: Settings | None = None) -> OpenAIEmbedder:
    """Build the embedder for the configured provider (LLM_PROVIDER in .env)."""
    settings = settings or get_settings()
    # max_retries=0: tenacity owns retrying, so the SDK's built-in retries don't multiply ours.
    if settings.llm_provider == "azure":
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": settings.azure_openai_embedding_deployment,
            }.items()
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Missing settings for Azure OpenAI: {', '.join(missing)}")
        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            max_retries=0,
            timeout=30,
        )
        model = settings.azure_openai_embedding_deployment
    else:
        if not settings.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to .env (see .env.example)."
            )
        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(), max_retries=0, timeout=30
        )
        model = settings.openai_embedding_model

    return OpenAIEmbedder(
        client,
        model=model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
