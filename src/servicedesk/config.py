"""Application settings, loaded from environment variables and an optional .env file.

pydantic-settings maps each field to an env var of the same name (case-insensitive),
validates types, and keeps secrets wrapped in SecretStr so they don't leak into logs.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"

    database_url: str = "postgresql+psycopg://servicedesk:servicedesk@localhost:5432/servicedesk"

    llm_provider: Literal["openai", "azure"] = "openai"

    # OpenAI
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # Embeddings. The dimension must match the vector(N) column in the database.
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100

    # Chunking (measured in tokens, the unit embedding models and LLMs work in)
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment once per process)."""
    return Settings()
