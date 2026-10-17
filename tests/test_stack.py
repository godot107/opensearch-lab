"""Integration checks against the docker compose stack; skipped when it's down."""
import pytest

from oslab import search, stack, store, sync

pytestmark = [pytest.mark.stack,
              pytest.mark.skipif(not stack.is_up(), reason="docker compose stack not running")]

CORPUS = {
    "a": {"title": "Stem cells", "text": "Mesenchymal stem cells repair cartilage."},
    "b": {"title": "Vaccines", "text": "Measles vaccination reduces child mortality."},
    "c": {"title": "", "text": "Cartilage repair in knees after injury."},
}


@pytest.fixture(scope="module")
def pg():
    conn = stack.postgres()
    conn.execute("create schema if not exists oslab_test")
    conn.execute("set search_path to oslab_test")
    conn.execute("drop table if exists docs")
    store.load(CORPUS, conn)
    yield conn
    conn.execute("drop schema oslab_test cascade")


@pytest.fixture(scope="module")
def os_index(pg):
    client, index = stack.opensearch(), "oslab-test-docs"
    sync.rebuild(index=index, os_client=client, pg_conn=pg)
    yield client, index
    client.indices.delete(index=index)


def test_load_is_idempotent(pg):
    assert store.load(CORPUS, pg) == 3
    assert pg.execute("select count(*) from docs").fetchone()[0] == 3


def test_pg_or_matches_any_word_and_and_needs_all(pg):
    assert set(search.pg("cartilage vaccination", conn=pg, mode="or")) == {"a", "b", "c"}
    assert search.pg("cartilage vaccination", conn=pg, mode="and") == []


def test_pg_query_text_is_data_not_sql(pg):
    assert search.pg("'); drop table docs; --", conn=pg) == []
    assert search.pg("the of and", conn=pg) == []  # all stopwords -> empty tsquery, no error
    assert pg.execute("select count(*) from docs").fetchone()[0] == 3


def test_bm25_stems_and_ranks(os_index):
    client, index = os_index
    hits = search.os_bm25("repairing cartilages", client=client, index=index)
    assert set(hits) == {"a", "c"}


def test_mapping_is_strict(os_index):
    client, index = os_index
    with pytest.raises(Exception):
        client.index(index=index, body={"title": "x", "body": "y", "surprise": 1})
