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
