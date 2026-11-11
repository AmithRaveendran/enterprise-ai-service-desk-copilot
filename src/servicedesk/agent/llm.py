"""The chat model the agent thinks with: LangChain's ChatOpenAI or AzureChatOpenAI.

Phase 3 calls the OpenAI SDK directly (one structured call). The agent uses LangChain's
wrapper instead because LangGraph works with LangChain messages and tool calling
(`bind_tools`, AIMessage.tool_calls, ToolMessage). Credentials are still checked by our own
openai_client.model_name(), so a missing setting gives the same clear ConfigurationError.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from servicedesk.config import Settings, get_settings
from servicedesk.openai_client import model_name


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    name = model_name(settings, "chat")  # raises ConfigurationError if anything is missing
    common = {
        "timeout": settings.llm_timeout_seconds,
        # The SDK retries rate limits, timeouts and 5xx with exponential backoff.
        # 2 retries: a person is waiting for the reply.
        "max_retries": 2,
        "reasoning_effort": settings.llm_reasoning_effort,
    }
    if settings.llm_provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=name,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            **common,
        )
    return ChatOpenAI(model=name, api_key=settings.openai_api_key, **common)
