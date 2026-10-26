from types import SimpleNamespace

import openai
import pytest
from tenacity import wait_none

from servicedesk.config import Settings
from servicedesk.ingestion.embeddings import ConfigurationError, OpenAIEmbedder, get_embedder


class FakeOpenAIClient:
    """Mimics client.embeddings.create(); can fail a set number of times first."""

    def __init__(self, failures: int = 0, error: Exception | None = None, reverse=False):
        self.calls: list[dict] = []
        self.failures = failures
        self.error = error
        self.reverse = reverse
        self.embeddings = self

    def create(self, model, input, dimensions):
        self.calls.append({"model": model, "input": input, "dimensions": dimensions})
        if self.failures:
            self.failures -= 1
            raise self.error
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(input))]
        if self.reverse:
            data.reverse()  # the API doesn't promise to return items in input order
        return SimpleNamespace(data=data)


class FakeConnectionError(openai.APIConnectionError):
    """A real APIConnectionError subclass that needs no HTTP request object to construct."""

    def __init__(self):
        Exception.__init__(self, "Connection error.")


def connection_error() -> openai.APIConnectionError:
    return FakeConnectionError()


def make_embedder(client, **kwargs) -> OpenAIEmbedder:
    return OpenAIEmbedder(client, model="m", dimensions=8, wait=wait_none(), **kwargs)


def test_texts_are_sent_in_batches():
    client = FakeOpenAIClient()
    vectors = make_embedder(client, batch_size=100).embed([f"t{i}" for i in range(250)])
    assert [len(c["input"]) for c in client.calls] == [100, 100, 50]
    assert len(vectors) == 250
    assert client.calls[0]["dimensions"] == 8


def test_vectors_follow_input_order_even_if_api_reorders():
    client = FakeOpenAIClient(reverse=True)
    assert make_embedder(client).embed(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]


def test_transient_errors_are_retried():
    client = FakeOpenAIClient(failures=2, error=connection_error())
    assert make_embedder(client).embed(["a"]) == [[0.0]]
    assert len(client.calls) == 3


def test_gives_up_after_max_attempts():
    client = FakeOpenAIClient(failures=10, error=connection_error())
    with pytest.raises(openai.APIConnectionError):
        make_embedder(client, max_attempts=3).embed(["a"])
    assert len(client.calls) == 3


def test_non_retryable_errors_fail_immediately():
    client = FakeOpenAIClient(failures=1, error=ValueError("bad request"))
    with pytest.raises(ValueError):
        make_embedder(client).embed(["a"])
    assert len(client.calls) == 1


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        get_embedder(Settings(_env_file=None, openai_api_key=None))


def test_azure_requires_its_settings():
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        get_embedder(Settings(_env_file=None, llm_provider="azure"))
