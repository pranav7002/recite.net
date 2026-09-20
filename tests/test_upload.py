"""Document upload tests. Embeddings are faked, so indexing needs no Gemini."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend import api, store
from tests.helpers import make_upload

TEXT = ("The CPU fetches instructions from memory and decodes them before execution. "
        "Registers hold operands close to the arithmetic logic unit. "
        "The program counter points to the next instruction in memory. "
        "Cache memory sits between the processor and main memory. "
        "Pipelining overlaps instruction execution to improve throughput.")


def test_duplicate_upload_one_document():
    r1 = api.upload(make_upload("notes.txt", TEXT.encode()))
    r2 = api.upload(make_upload("notes.txt", TEXT.encode()))
    assert r1["doc_id"] == r2["doc_id"]
    assert len(store.list_documents()) == 1


def test_upload_unsupported_type_415():
    with pytest.raises(HTTPException) as exc:
        api.upload(make_upload("photo.png", b"not a pdf"))
    assert exc.value.status_code == 415


def test_scanned_pdf_422():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()
    with pytest.raises(HTTPException) as exc:
        api.upload(make_upload("scan.pdf", pdf))
    assert exc.value.status_code == 422
