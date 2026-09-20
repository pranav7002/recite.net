"""Pending-action confirmation tests."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend import api, guardrails, store, tools


def test_confirm_after_10_minutes_rejected(db):
    pid = store.save_pending("email_summary", {"to": "friend@uni.edu", "subject": "s", "body": "b"})
    with db.cursor() as cur:
        cur.execute(
            "UPDATE pending SET created_at = now() - interval '11 minutes' WHERE id = %s",
            (pid,),
        )
    assert store.confirm_pending(pid) is None


def test_confirm_is_single_execution(db):
    pid = store.save_pending("email_summary", {"to": "friend@uni.edu", "subject": "s", "body": "b"})
    assert store.confirm_pending(pid) is not None
    assert store.confirm_pending(pid) is None


def test_cancel_pending(db):
    pid = store.save_pending("email_summary", {"to": "friend@uni.edu", "subject": "s", "body": "b"})
    assert store.cancel_pending(pid) is True
    assert store.confirm_pending(pid) is None


def test_confirm_second_returns_409(db, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset({"friend@uni.edu"}))
    pid = store.save_pending("email_summary", {"to": "friend@uni.edu", "subject": "s", "body": "b"})
    api.confirm(pid)
    with pytest.raises(HTTPException) as exc:
        api.confirm(pid)
    assert exc.value.status_code == 409


def test_confirm_recipient_removed_409(db, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset())
    pid = store.save_pending("email_summary", {"to": "friend@uni.edu", "subject": "s", "body": "b"})
    with pytest.raises(HTTPException) as exc:
        api.confirm(pid)
    assert exc.value.status_code == 409
    with db.cursor() as cur:
        cur.execute("SELECT status FROM pending WHERE id = %s", (pid,))
        assert cur.fetchone()[0] == "cancelled"


def test_confirm_failed_write_marks_failed(db, monkeypatch):
    monkeypatch.setattr(guardrails, "CONTACTS", frozenset({"friend@uni.edu"}))
    pid = store.save_pending("email_summary", {"to": "friend@uni.edu", "subject": "s", "body": "b"})

    def boom(name, args):
        raise OSError("disk full")

    monkeypatch.setattr(tools, "run_tool", boom)
    with pytest.raises(HTTPException) as exc:
        api.confirm(pid)
    assert exc.value.status_code == 500
    with db.cursor() as cur:
        cur.execute("SELECT status FROM pending WHERE id = %s", (pid,))
        assert cur.fetchone()[0] == "failed"
