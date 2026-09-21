"""Chunking tests (ingest.chunk_pages)."""
from __future__ import annotations

from backend import ingest


def _page(*paragraph_texts: str) -> ingest.Page:
    return ingest.Page(page=1, paragraphs=[
        ingest.Paragraph(page=1, text=t, section=None) for t in paragraph_texts
    ])


def test_chunk_pages_no_empty_chunk():
    chunks = ingest.chunk_pages([_page("word " * 600, "short.")])
    assert chunks
    assert all(c.text.strip() for c in chunks)


def test_chunk_pages_overlap_from_long_paragraph():
    tail = "zebra"
    long_text = "filler " * 600 + tail
    short_text = "second paragraph."
    chunks = ingest.chunk_pages([_page(long_text, short_text)])
    short_chunk = next(c for c in chunks if short_text in c.text)
    assert tail in short_chunk.text
    assert short_chunk.text.index(tail) < short_chunk.text.index(short_text)


def _numbered_page(n: int, text: str) -> ingest.Page:
    return ingest.Page(page=n, paragraphs=[ingest.Paragraph(page=n, text=text, section=None)])


def test_split_pages_chunks_never_span_pages():
    pages = [_numbered_page(1, "alpha one."), _numbered_page(2, "beta two."), _numbered_page(3, "gamma three.")]
    chunks = ingest.chunk_pages(pages, split_pages=True)
    assert [c.page for c in chunks] == [1, 2, 3]
    assert "alpha" not in chunks[1].text and "beta" not in chunks[2].text   # no overlap across pages


def test_default_chunking_may_span_pages():
    pages = [_numbered_page(1, "alpha one."), _numbered_page(2, "beta two.")]
    chunks = ingest.chunk_pages(pages)
    assert len(chunks) == 1 and chunks[0].page == 1
