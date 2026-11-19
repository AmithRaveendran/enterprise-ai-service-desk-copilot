from servicedesk.config import Settings


def test_defaults_load_without_env_file():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.database_url.startswith("postgresql+psycopg://")


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-prod")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "azure"
    assert settings.azure_openai_chat_deployment == "chat-prod"


def test_secrets_are_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    settings = Settings(_env_file=None)
    assert "sk-test" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-test-not-a-real-key"


def test_the_chat_model_is_built_from_settings_without_a_network_call():
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    from servicedesk.agent.llm import get_chat_model

    openai = get_chat_model(Settings(_env_file=None, openai_api_key="sk-test"))
    assert isinstance(openai, ChatOpenAI) and openai.model_name == "gpt-5-mini"
    assert openai.reasoning_effort == "low" and openai.max_retries == 2

    azure = get_chat_model(
        Settings(
            _env_file=None,
            llm_provider="azure",
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="key",
            azure_openai_chat_deployment="chat-deployment",
        )
    )
    assert isinstance(azure, AzureChatOpenAI)
    assert azure.deployment_name == "chat-deployment"
