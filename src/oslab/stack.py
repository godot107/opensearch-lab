"""Connections to the local docker compose stack. Override with OSLAB_OS_URL / OSLAB_PG_DSN."""
import os

import psycopg
from opensearchpy import OpenSearch

OS_URL = os.environ.get("OSLAB_OS_URL", "http://localhost:9200")
PG_DSN = os.environ.get("OSLAB_PG_DSN", "postgresql://oslab:oslab@localhost:5432/oslab")


def opensearch():
    return OpenSearch(OS_URL, timeout=60)


def postgres():
    return psycopg.connect(PG_DSN, autocommit=True)


def is_up():
    try:
        return bool(opensearch().ping()) and postgres().execute("select 1").fetchone() == (1,)
    except Exception:
        return False
