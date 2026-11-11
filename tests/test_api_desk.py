"""REST endpoints for service status and tickets."""

import pytest
from fastapi.testclient import TestClient

from servicedesk.api.app import app
from servicedesk.api.dependencies import get_desk

pytestmark = pytest.mark.integration

NEW_TICKET = {
    "title": "Keyboard keys sticking",
    "description": "Several keys on my laptop keyboard stick.",
    "category": "hardware",
    "requester": "li.wei@halcyon-dynamics.example",
}


@pytest.fixture
def client(desk):
    app.dependency_overrides[get_desk] = lambda: desk
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_all_service_status(client):
    body = client.get("/services/status").json()
    assert len(body["services"]) == 10


def test_one_service_status(client):
    assert client.get("/services/wifi/status").json()["name"] == "Wi-Fi"
    response = client.get("/services/fax/status")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found" and "Known services" in response.json()["detail"]


def test_create_ticket_is_201(client):
    response = client.post("/tickets", json=NEW_TICKET)
    assert response.status_code == 201
    assert response.json()["number"] == "INC-1006"


def test_create_ticket_validation_is_422(client):
    response = client.post("/tickets", json=NEW_TICKET | {"requester": "li@gmail.com"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "requester"]


def test_search_tickets_with_query_parameters(client):
    params = {"requester": "priya.sharma@halcyon-dynamics.example", "open_only": "true"}
    body = client.get("/tickets", params=params).json()
    assert body["count"] == 2


def test_search_rejects_unknown_and_invalid_parameters(client):
    assert client.get("/tickets", params={"limit": 500}).status_code == 422
    assert client.get("/tickets", params={"colour": "red"}).status_code == 422


@pytest.mark.parametrize("ticket_id", ["1001", "INC-1001"])
def test_get_ticket_by_number(client, ticket_id):
    assert client.get(f"/tickets/{ticket_id}").json()["id"] == 1001


def test_get_missing_ticket_is_404(client):
    assert client.get("/tickets/9999").status_code == 404
    assert client.get("/tickets/abc").status_code == 422


def test_patch_ticket(client):
    response = client.patch("/tickets/1003", json={"status": "in_progress"})
    assert response.status_code == 200 and response.json()["status"] == "in_progress"


def test_patch_rules(client):
    assert client.patch("/tickets/1003", json={}).status_code == 422  # nothing to change
    assert client.patch("/tickets/1003", json={"status": "resolved"}).status_code == 422
    response = client.patch("/tickets/1005", json={"priority": "P4"})  # closed
    assert response.status_code == 409 and response.json()["code"] == "conflict"
    assert client.patch("/tickets/1003", json={"status": "closed"}).status_code == 409


def test_add_comment_is_201_and_returns_the_ticket(client):
    comment = {"author": "tom.oconnor@halcyon-dynamics.example", "body": "Safe mode works."}
    response = client.post("/tickets/1002/comments", json=comment)
    assert response.status_code == 201
    assert response.json()["status"] == "in_progress"  # requester replied


def test_openapi_lists_the_desk_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/services/status", "/tickets", "/tickets/{ticket_id}"} <= paths.keys()
