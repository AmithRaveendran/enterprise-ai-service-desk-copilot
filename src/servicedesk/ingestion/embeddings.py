"""Turn text into embedding vectors with OpenAI or Azure OpenAI.

An embedding is a list of numbers (1536 for text-embedding-3-small) that captures what a
text *means*. Texts with similar meanings get vectors that point in similar directions, so
"I can't log in" lands near "Resetting your password" even with no words in common.
"""

from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI
from tenacity.wait import wait_base

from servicedesk.config import Settings, get_settings

# Client building and the retry policy are shared with the chat model; re-exported here so
# `from servicedesk.ingestion.embeddings import ConfigurationError` keeps working.
from servicedesk.openai_client import (
    RETRYABLE_ERRORS,
    ConfigurationError,
    client_and_model,
    is_retryable,
    make_retrying,
)

__all__ = ["RETRYABLE_ERRORS", "ConfigurationError", "Embedder", "OpenAIEmbedder", "is_retryable"]


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
        self._retrying = make_retrying(max_attempts, wait)

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
    client, model = client_and_model(settings, "embedding")
    return OpenAIEmbedder(
        client,
        model=model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
