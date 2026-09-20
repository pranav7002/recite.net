"""Shared fixtures: a rolled-back DB transaction, fake embeddings, and isolated
file paths so tests never touch Gemini or the real outbox/uploads."""
from __future__ import annotations

import pytest

from backend import store


@pytest.fixture(scope="session", autouse=True)
def _schema():
    store.init_db()


@pytest.fixture(autouse=True)
def db():
    with store.test_transaction() as conn:
        yield conn


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    def fake_embed(texts, model=None, batch_size=None, dimensions=768, task_type=None):
        return [[0.01] * 768 for _ in texts]

    monkeypatch.setattr("backend.llm.embed", fake_embed)


@pytest.fixture(autouse=True)
def outbox(tmp_path, monkeypatch):
    path = tmp_path / "outbox.json"
    monkeypatch.setattr("backend.tools.OUTBOX_PATH", path)
    return path


@pytest.fixture(autouse=True)
def uploads(tmp_path, monkeypatch):
    d = tmp_path / "uploads"
    monkeypatch.setattr("backend.api.UPLOADS_DIR", d)
    return d
