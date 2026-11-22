"""API key authentication, the production start-up check, and request IDs.

None of these need the database: auth runs before the route does, so a rejected request
never reaches it, and an accepted one is shown by the route's own 422 for a missing query
parameter (FastAPI checks router dependencies first, then the parameters).
"""

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from servicedesk.api import app as app_module
from servicedesk.api.app import app
from servicedesk.api.dependencies import get_session_factory
from servicedesk.api.security import MIN_KEY_LENGTH, check_production_settings
from servicedesk.config import Settings, get_settings

KEY = "k" * 32


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)  # ignore the developer's .env


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def with_key(client):
    app.dependency_overrides[get_settings] = lambda: settings(api_key=SecretStr(KEY))
    return client


def test_requests_are_open_when_no_key_is_configured(client):
    app.dependency_overrides[get_settings] = lambda: settings(api_key=None)
    assert client.get("/search").status_code == 422  # reached the route: `q` is missing


def test_missing_key_is_401(with_key):
    response = with_key.get("/search")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "APIKey"


def test_wrong_key_is_401(with_key):
    assert with_key.get("/search", headers={"X-API-Key": "nope"}).status_code == 401


def test_right_key_is_accepted(with_key):
    assert with_key.get("/search", headers={"X-API-Key": KEY}).status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/ask"),
        ("get", "/services/status"),
        ("get", "/tickets"),
        ("post", "/tickets"),
        ("post", "/chat"),
        ("get", "/chat/some-thread"),
    ],
)
def test_every_router_is_protected(with_key, method, path):
    assert getattr(with_key, method)(path).status_code == 401


def test_health_needs_no_key(with_key):
    class DeadFactory:  # health reports the database as unreachable, but answers
        def __call__(self):
            from sqlalchemy.exc import OperationalError

            raise OperationalError("SELECT 1", {}, Exception("down"))

    app.dependency_overrides[get_session_factory] = DeadFactory
    response = with_key.get("/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_docs_show_the_api_key_scheme(client):
    schemes = client.get("/openapi.json").json()["components"]["securitySchemes"]
    assert schemes["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "The API key set in API_KEY.",
    }


# --- production start-up check ---


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"app_env": "prod", "api_key": None}, "API_KEY must be set"),
        ({"app_env": "prod", "api_key": SecretStr("short")}, f"at least {MIN_KEY_LENGTH}"),
    ],
)
def test_production_refuses_unsafe_settings(values, error):
    with pytest.raises(RuntimeError, match=error):
        check_production_settings(settings(**values))


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_api_key_means_no_key(blank):
    # `API_KEY=` in .env must switch authentication off, not demand an empty key.
    assert settings(api_key=blank).api_key is None


def test_production_accepts_a_long_key_and_dev_needs_none():
    check_production_settings(settings(app_env="prod", api_key=SecretStr(KEY)))
    check_production_settings(settings(app_env="dev", api_key=None))


def test_app_does_not_start_in_production_without_a_key(monkeypatch):
    monkeypatch.setattr(app_module, "get_settings", lambda: settings(app_env="prod", api_key=None))
    with pytest.raises(RuntimeError, match="API_KEY must be set"), TestClient(app):
        pass  # entering the client runs the app's startup


# --- request IDs and the request log ---


def test_a_request_id_is_generated_and_logged(client, caplog):
    app.dependency_overrides[get_settings] = lambda: settings(api_key=None)
    with caplog.at_level(logging.INFO, logger="servicedesk.api"):
        response = client.get("/search")
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32
    (line,) = [r.getMessage() for r in caplog.records if r.name == "servicedesk.api"]
    assert f"request_id={request_id} method=GET path=/search status=422" in line
    assert "duration_ms=" in line


def test_a_valid_incoming_request_id_is_reused(client):
    response = client.get("/openapi.json", headers={"X-Request-ID": "gateway-123.a_b"})
    assert response.headers["x-request-id"] == "gateway-123.a_b"


def test_an_invalid_incoming_request_id_is_replaced(client):
    response = client.get("/openapi.json", headers={"X-Request-ID": "bad id; drop table"})
    assert response.headers["x-request-id"] != "bad id; drop table"
    assert len(response.headers["x-request-id"]) == 32
