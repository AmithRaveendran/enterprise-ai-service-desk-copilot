import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from servicedesk.api import dependencies
from servicedesk.api.app import app
from servicedesk.api.dependencies import get_answerer, get_retriever, get_session_factory
from servicedesk.rag.answerer import Answerer
from servicedesk.search.retriever import Retriever
from tests.conftest import FakeEmbedder, FakeLLM
from tests.test_embeddings import connection_error

pytestmark = pytest.mark.integration


@pytest.fixture
def client(kb_session_factory):
    """The real app, pointed at the test database and a fake embedder."""
    app.dependency_overrides[get_session_factory] = lambda: kb_session_factory
    app.dependency_overrides[get_retriever] = lambda: Retriever(kb_session_factory, FakeEmbedder())
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_database_and_counts(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "documents": 19, "chunks": 108}


def test_health_is_503_when_database_is_down(client):
    dead = create_engine(
        "postgresql+psycopg://x:x@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    app.dependency_overrides[get_session_factory] = lambda: sessionmaker(bind=dead)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_search_returns_ranked_results_with_citation_fields(client):
    response = client.get("/search", params={"q": "Secure Print", "k": 3, "mode": "keyword"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "keyword" and body["count"] == 3
    first = body["results"][0]
    assert first["rank"] == 1 and first["doc_id"] == "KB-0006"
    assert {"title", "source_path", "section", "content", "score"} <= first.keys()


def test_search_defaults_to_hybrid(client):
    body = client.get("/search", params={"q": "Teams audio docking station"}).json()
    assert body["mode"] == "hybrid"
    assert body["count"] == 5
    assert body["results"][0]["doc_id"] == "KI-0012"


def test_search_filters(client):
    params = {"q": "password", "category": "policy", "doc_type": "pdf", "k": 10}
    results = client.get("/search", params=params).json()["results"]
    assert results and all(r["doc_id"].startswith("POL-") for r in results)


@pytest.mark.parametrize(
    "params",
    [
        {},  # q missing
        {"q": "a"},  # too short
        {"q": "x" * 501},  # too long
        {"q": "vpn", "k": 0},
        {"q": "vpn", "k": 21},
        {"q": "vpn", "mode": "fuzzy"},
        {"q": "vpn", "doc_type": "docx"},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, params):
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][0] == "query"  # says which parameter is wrong


def test_vector_search_without_api_key_is_503(client, kb_session_factory):
    app.dependency_overrides[get_retriever] = lambda: Retriever(kb_session_factory, None)
    assert client.get("/search", params={"q": "vpn", "mode": "vector"}).status_code == 503
    assert client.get("/search", params={"q": "vpn", "mode": "keyword"}).status_code == 200


def test_openapi_schema_documents_the_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/search", "/ask"} <= paths.keys()


# --- POST /ask ----------------------------------------------------------------------------


def use_llm(kb_session_factory, llm, min_similarity=0.0):
    """Point /ask at the test KB and a fake model. The fake embedder's similarities aren't
    on OpenAI's scale, so the gate is off unless a test sets it."""
    retriever = Retriever(kb_session_factory, FakeEmbedder())
    app.dependency_overrides[get_answerer] = lambda: Answerer(
        retriever, llm, min_similarity=min_similarity
    )


def test_ask_returns_an_answer_with_citations(client, kb_session_factory):
    use_llm(kb_session_factory, FakeLLM("Release the job at the printer [2]."))
    response = client.post("/ask", json={"question": "How does Secure Print work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False and body["refusal_reason"] is None
    assert body["answer"] == "Release the job at the printer [1]."
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["number"] == 1 and citation["content"]
    assert {"doc_id", "title", "section", "source_path", "chunk_id"} <= citation.keys()
    assert body["model"] == "fake-llm" and body["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
    }


def test_ask_refuses_weak_evidence(client, kb_session_factory):
    llm = FakeLLM()
    use_llm(kb_session_factory, llm, min_similarity=0.99)
    body = client.post("/ask", json={"question": "capital of France"}).json()
    assert body["refused"] is True and body["refusal_reason"] == "no_relevant_sources"
    assert body["citations"] == [] and body["model"] is None
    assert llm.prompts == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "hi"},  # too short
        {"question": "x" * 1001},
        {"question": "vpn drops", "doc_type": "docx"},
    ],
)
def test_ask_rejects_invalid_requests(client, payload):
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][0] == "body"


def test_ask_without_chat_api_key_is_503(client, monkeypatch):
    monkeypatch.setattr(dependencies, "_llm", lambda: None)
    response = client.post("/ask", json={"question": "vpn drops"})
    assert response.status_code == 503


def test_ask_is_502_when_the_model_api_fails(client, kb_session_factory):
    use_llm(kb_session_factory, FakeLLM(error=connection_error()))
    response = client.post("/ask", json={"question": "How does Secure Print work?"})
    assert response.status_code == 502
