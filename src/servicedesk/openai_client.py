"""One place to build OpenAI / Azure OpenAI clients and to decide which errors to retry.

Both the embedder (ingestion, search) and the chat model (answers) use this, so switching
LLM_PROVIDER in .env switches everything at once.
"""

from typing import Literal

import openai
from openai import AzureOpenAI, OpenAI
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from tenacity.wait import wait_base

from servicedesk.config import Settings

# Errors worth retrying: they are usually temporary. Others, such as a bad API key
# (AuthenticationError) or an invalid request, fail immediately.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class ConfigurationError(RuntimeError):
    pass


def is_retryable(exc: BaseException) -> bool:
    # Running out of credit also arrives as a 429 RateLimitError, but waiting won't fix it.
    if isinstance(exc, openai.RateLimitError) and exc.code == "insufficient_quota":
        return False
    return isinstance(exc, RETRYABLE_ERRORS)


def make_retrying(max_attempts: int = 5, wait: wait_base | None = None) -> Retrying:
    """Exponential backoff with jitter: wait ~1s, 2s, 4s... plus randomness, so many clients
    hitting a rate limit don't all retry at the same instant."""
    return Retrying(
        retry=retry_if_exception(is_retryable),
        wait=wait or wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )


def model_name(settings: Settings, kind: Literal["chat", "embedding"]) -> str:
    """Check the credentials for LLM_PROVIDER and return what to call: a model name for
    OpenAI, a deployment name for Azure. Raises ConfigurationError naming any missing setting.
    Shared by our own clients and the LangChain chat model the agent uses."""
    if settings.llm_provider == "azure":
        deployment_setting = f"AZURE_OPENAI_{kind.upper()}_DEPLOYMENT"
        deployment = (
            settings.azure_openai_chat_deployment
            if kind == "chat"
            else settings.azure_openai_embedding_deployment
        )
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
                deployment_setting: deployment,
            }.items()
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Missing settings for Azure OpenAI: {', '.join(missing)}")
        return deployment

    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
    return settings.openai_chat_model if kind == "chat" else settings.openai_embedding_model


def client_and_model(
    settings: Settings, kind: Literal["chat", "embedding"], timeout: float = 30
) -> tuple[OpenAI, str]:
    """The client for LLM_PROVIDER and the model (or Azure deployment) to call."""
    model = model_name(settings, kind)
    # max_retries=0: tenacity owns retrying, so the SDK's built-in retries don't multiply ours.
    if settings.llm_provider == "azure":
        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            max_retries=0,
            timeout=timeout,
        )
        return client, model

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value(), max_retries=0, timeout=timeout
    )
    return client, model
