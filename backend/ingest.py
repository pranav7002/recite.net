"""Index one document into the store.

index_document() is a plain function taking a path because Day 2's upload
endpoint calls it unchanged.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from backend import llm, store

MIN_TEXT_CHARS = 200
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
DOC_ID_CHARS = 12

# tiktoken has no Gemini tokenizer; cl100k_base is the standard approximation.
_ENC = tiktoken.get_encoding("cl100k_base")

# A span counts as a heading if it is at least this much larger than body text.
HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 100


class UnsupportedFile(Exception):
    def __init__(self, suffix: str) -> None:
        super().__init__(f"Unsupported file type: {suffix or '(none)'}")


class NoTextLayer(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"No extractable text in {name} — it may be a scanned PDF")


@dataclass
class Chunk:
    id: str            # f"{doc_id}:{n}"
    doc_id: str        # sha256 of the file bytes, first 12 chars
    doc_name: str
    page: int          # 1-based, for citations
    section: str | None
    text: str


@dataclass
class Paragraph:
    page: int
    text: str
    section: str | None


@dataclass
class Page:
    page: int
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


def extract(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md"}:
        return extract_text(path)
    raise UnsupportedFile(path.suffix)


# ---- text files ------------------------------------------------------------


def extract_text(path: Path) -> list[Page]:
    """Split on headings; a chunk's page is its section's 1-based index."""
    paragraphs: list[Paragraph] = []
    section: str | None = None
    section_index = 0
    buf: list[str] = []

    def flush_body() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if para:
                paragraphs.append(Paragraph(page=max(section_index, 1), text=para, section=section))

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        heading = _txt_heading(raw)
        if heading is None:
            buf.append(raw)
        else:
            flush_body()
            section = heading
            section_index += 1
            paragraphs.append(Paragraph(page=section_index, text=heading, section=heading))
    flush_body()
    return _pages_from_paragraphs(paragraphs)


def _txt_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):                       # markdown heading
        return stripped.lstrip("#").strip() or None
    if len(stripped) <= 80:                            # short ALL-CAPS line
        letters = [c for c in stripped if c.isalpha()]
        if len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
            return stripped
    return None


def _pages_from_paragraphs(paragraphs: list[Paragraph]) -> list[Page]:
    pages: list[Page] = []
    for para in paragraphs:
        if pages and pages[-1].page == para.page:
            pages[-1].paragraphs.append(para)
        else:
            pages.append(Page(page=para.page, paragraphs=[para]))
    return pages


# ---- PDF -------------------------------------------------------------------


def extract_pdf(path: Path) -> list[Page]:
    import pymupdf

    doc = pymupdf.open(path)
    try:
        pages: list[Page] = []
        section: str | None = None
        for page_no, page in enumerate(doc, start=1):
            body_size = _body_font_size(page)
            paragraphs, section = _pdf_paragraphs(page, page_no, body_size, section)
            if paragraphs:
                pages.append(Page(page=page_no, paragraphs=paragraphs))
        return pages
    finally:
        doc.close()


def _body_font_size(page) -> float:
    """Weighted median span size — robust to a few large heading spans."""
    pairs = sorted(
        (s["size"], max(len(s.get("text", "").strip()), 1))
        for block in page.get_text("dict")["blocks"] if block.get("type") == 0
        for line in block.get("lines", [])
        for s in line.get("spans", []) if s.get("text", "").strip()
    )
    if not pairs:
        return 0.0
    total = sum(w for _, w in pairs)
    seen = 0
    for size, weight in pairs:
        seen += weight
        if seen >= total / 2:
            return size
    return pairs[-1][0]


def _pdf_paragraphs(page, page_no: int, body_size: float, section: str | None) -> tuple[list[Paragraph], str | None]:
    paragraphs: list[Paragraph] = []
    buf: list[str] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if not line_text:
                continue
            if any(_is_heading_span(s.get("text", "").strip(), s.get("size", 0.0), body_size) for s in spans):
                if buf:
                    paragraphs.append(Paragraph(page=page_no, text="\n".join(buf), section=section))
                    buf = []
                section = line_text
                paragraphs.append(Paragraph(page=page_no, text=line_text, section=section))
            else:
                buf.append(line_text)
    if buf:
        paragraphs.append(Paragraph(page=page_no, text="\n".join(buf), section=section))
    return paragraphs, section


def _is_heading_span(text: str, size: float, body_size: float) -> bool:
    if body_size <= 0 or not text or len(text) > HEADING_MAX_CHARS:
        return False
    if re.fullmatch(r"[\d\s.\-/:]+", text):            # page numbers are not headings
        return False
    return size >= body_size * HEADING_SIZE_RATIO and size >= body_size + 0.5


# ---- chunking --------------------------------------------------------------


def chunk_pages(pages: list[Page], size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[Chunk]:
    """Paragraph-boundary chunks of ~size tokens with overlap.

    A chunk's page and section come from its first fresh paragraph, so a chunk
    that spans pages records the page where it starts. Paragraphs longer than
    `size` fall back to hard token splits. doc fields are filled in by
    index_document().
    """
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_page = 0
    buf_section: str | None = None
    buf_fresh = False

    def finalize() -> None:
        nonlocal buf, buf_tokens, buf_fresh
        text = "\n\n".join(buf)
        chunks.append(Chunk(id="", doc_id="", doc_name="", page=buf_page, section=buf_section, text=text))
        carry = _ENC.decode(_ENC.encode(text)[-overlap:]) if overlap else ""
        buf, buf_tokens, buf_fresh = [], 0, False
        if carry:
            buf.append(carry)
            buf_tokens = len(_ENC.encode(carry))

    def add(text: str, page: int, section: str | None) -> None:
        nonlocal buf_tokens, buf_page, buf_section, buf_fresh
        if not buf_fresh:
            buf_page, buf_section = page, section
        buf.append(text)
        buf_tokens += len(_ENC.encode(text))
        buf_fresh = True

    for page in pages:
        for para in page.paragraphs:
            tokens = len(_ENC.encode(para.text))
            if tokens > size:                          # oversized paragraph: hard split
                if buf_fresh:                          # only flush if there is real, unsaved text
                    finalize()
                encoded = _ENC.encode(para.text)
                for i in range(0, len(encoded), size - overlap):
                    chunks.append(Chunk(id="", doc_id="", doc_name="", page=para.page,
                                        section=para.section, text=_ENC.decode(encoded[i : i + size])))
                # carry overlap from the end of THIS paragraph, not from text before it
                tail = _ENC.decode(encoded[-overlap:]) if overlap else ""
                buf = [tail] if tail else []
                buf_tokens = len(_ENC.encode(tail)) if tail else 0
                buf_fresh = False
                continue
            if buf_fresh and buf_tokens + tokens > size:
                finalize()
            elif not buf_fresh and buf_tokens + tokens > size:
                buf, buf_tokens = [], 0                # drop the carry, start clean
            add(para.text, para.page, para.section)
    if buf_fresh:
        finalize()
    return chunks


# ---- entry point -----------------------------------------------------------


def index_document(path: Path) -> list[Chunk]:
    """Index one document; re-uploading identical content is a no-op."""
    data = path.read_bytes()
    doc_id = hashlib.sha256(data).hexdigest()[:DOC_ID_CHARS]
    if store.has(doc_id):
        return store.get_chunks(doc_id)

    pages = extract(path)
    if sum(len(p.text) for p in pages) < MIN_TEXT_CHARS:
        raise NoTextLayer(path.name)

    chunks = chunk_pages(pages)
    for i, chunk in enumerate(chunks):
        chunk.id = f"{doc_id}:{i}"
        chunk.doc_id = doc_id
        chunk.doc_name = path.name

    vectors = llm.embed([c.text for c in chunks])
    full_text = "\n\n".join(f"[[page {p.page}]]\n{p.text}" for p in pages)
    store.add(doc_id=doc_id, name=path.name, full_text=full_text,
              chunks=chunks, vectors=llm.normalise(vectors))
    return chunks
