"""API authentication: a shared secret in the `X-API-Key` header.

Every route except /health depends on require_api_key. Health stays open because load
balancers and Docker health checks call it without credentials, and it reveals nothing
beyond "up" and two counts.

- No API_KEY configured (local development): every request is allowed.
- API_KEY configured: a request without the header, or with a wrong key, gets 401.
- APP_ENV=prod: check_production_settings() makes the app refuse to start without a key,
  so a deployment can't be left open by forgetting one variable ("secure by default").

A single shared key is the simplest scheme that is still real protection for
service-to-service calls (e.g. a chat front end calling this API). Per-user identity would
come from an identity provider (OAuth2/OIDC tokens checked here instead); see
docs/security.md.
"""

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from servicedesk.config import Settings, get_settings

MIN_KEY_LENGTH = 16  # short keys can be guessed; generate one with `secrets.token_urlsafe(32)`

# auto_error=False: a missing header gives None instead of FastAPI's own 403, so we decide
# (allow in dev, 401 otherwise). Declaring it as a security scheme also adds the
# "Authorize" button to /docs.
api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="The API key set in API_KEY."
)


def require_api_key(
    key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value().encode()
    # compare_digest takes the same time wherever the first difference is, so response
    # times don't leak how much of a guessed key was right (a "timing attack").
    if key is None or not hmac.compare_digest(key.encode(), expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid API key (X-API-Key header).",
            headers={"WWW-Authenticate": "APIKey"},
        )


def check_production_settings(settings: Settings) -> None:
    """Raise if the settings are unsafe to serve in production. Called at app startup."""
    if settings.app_env != "prod":
        return
    if settings.api_key is None:
        raise RuntimeError("API_KEY must be set when APP_ENV=prod.")
    if len(settings.api_key.get_secret_value()) < MIN_KEY_LENGTH:
        raise RuntimeError(f"API_KEY must be at least {MIN_KEY_LENGTH} characters.")
