CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,        -- sha256 of file bytes, first 12 chars
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_text (
    doc_id      TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    text        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,        -- "<doc_id>:<n>"
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INT  NOT NULL,           -- 1-based, for citations
    section     TEXT,
    text        TEXT NOT NULL,
    embedding   vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops);
