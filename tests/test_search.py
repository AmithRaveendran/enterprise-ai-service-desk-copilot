import pytest

from servicedesk.ingestion.embeddings import ConfigurationError
from servicedesk.search.retriever import Retriever, SearchFilters, reciprocal_rank_fusion
from tests.conftest import FakeEmbedder


def test_rrf_rewards_items_found_by_both_rankings():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 4]], k=60)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)  # 3rd in one list, 1st in the other
    assert scores[1] == pytest.approx(1 / 61)
    assert max(scores, key=scores.get) == 3


def test_rrf_with_one_ranking_keeps_its_order():
    scores = reciprocal_rank_fusion([[7, 5, 9]])
    assert sorted(scores, key=scores.get, reverse=True) == [7, 5, 9]


@pytest.fixture(scope="module")
def retriever(kb_session_factory):
    return Retriever(kb_session_factory, FakeEmbedder())


@pytest.mark.integration
def test_keyword_search_finds_exact_terms(retriever):
    hits = retriever.search("Secure Print", k=3, mode="keyword")
    assert [h.doc_id for h in hits] == ["KB-0006"] * 3
    assert all(h.keyword_rank and h.vector_rank is None for h in hits)


@pytest.mark.integration
def test_keyword_search_matches_word_forms(retriever):
    # "printers" is stemmed to 'printer', the same form stored in search_vector
    assert retriever.search("printers", k=1, mode="keyword")[0].doc_id == "KB-0006"


@pytest.mark.integration
def test_keyword_search_matches_any_word_not_all(retriever):
    hits = retriever.search("why does my securelink vpn keep disconnecting", k=3, mode="keyword")
    assert hits and hits[0].doc_id in {"KI-0007", "KB-0001"}


@pytest.mark.integration
def test_keyword_search_with_no_matching_words_returns_nothing(retriever):
    assert retriever.search("zzqx wqpv", mode="keyword") == []


@pytest.mark.integration
def test_vector_search_ranks_by_similarity(retriever):
    hits = retriever.search("SecureLink gateway unreachable captive portal", k=5, mode="vector")
    assert hits[0].doc_id == "KB-0001"
    similarities = [h.similarity for h in hits]
    assert similarities == sorted(similarities, reverse=True)
    assert all(-1 <= s <= 1 for s in similarities)


@pytest.mark.integration
def test_hybrid_merges_both_rankings(retriever):
    hits = retriever.search("Teams audio docking station", k=5, mode="hybrid")
    assert len(hits) == 5
    assert len({h.chunk_id for h in hits}) == 5  # no duplicates after fusion
    assert hits[0].doc_id == "KI-0012"
    assert hits[0].vector_rank and hits[0].keyword_rank  # found by both methods
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_filters_restrict_results(retriever, mode):
    hits = retriever.search(
        "password laptop software",
        k=10,
        mode=mode,
        filters=SearchFilters(category="policy", doc_type="pdf"),
    )
    assert hits
    assert all(h.doc_id.startswith("POL-") for h in hits)
    assert all(h.source_path.endswith(".pdf") for h in hits)


@pytest.mark.integration
def test_results_carry_citation_fields(retriever):
    hit = retriever.search("data classification levels", k=1, mode="hybrid")[0]
    assert hit.doc_id == "POL-0005"
    assert hit.title and hit.source_path and hit.section and hit.content
    assert hit.page in {1, 2}


@pytest.mark.integration
def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


@pytest.mark.integration
def test_vector_search_without_embedder_is_a_clear_error(kb_session_factory):
    keyword_only = Retriever(kb_session_factory, embedder=None)
    assert keyword_only.search("printer", mode="keyword")
    with pytest.raises(ConfigurationError):
        keyword_only.search("printer", mode="vector")
