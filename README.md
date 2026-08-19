# RAG-Document-QA

A FastAPI service that ingests documents, retrieves relevant passages by meaning, and answers questions grounded in the retrieved text — refusing rather than guessing when nothing relevant is found.

This is the semantic-search successor to [DocumentStore](https://github.com/sarmadnagori/document-store), which used keyword matching. See that repo's README for why keyword search wasn't enough.

---

## How it works

**Ingestion.** A document is split into paragraph-sized chunks. Each chunk is embedded once, at ingestion time, using `nomic-embed-text` — a 768-number vector representing its meaning. Both the text and the embedding are stored, one row per chunk.

**Retrieval.** The question is embedded with the same model. Every stored chunk is compared against it using cosine similarity, producing a relevance score per chunk. The top three, ranked by score, are selected.

**Generation.** The three chunks are assembled into a prompt with the question and sent to `llama3.2`. The model is instructed to answer only from the provided text, and to say so explicitly if the answer isn't there.

**Storage note.** Embeddings are stored as `TEXT` and compared in a Python loop rather than in the database. That's a deliberate simplification at this scale — every search scans the full table. A production system would use `pgvector` to store vectors natively and let Postgres do the ranking with an indexed query, avoiding the full scan.

---

## Why two refusal mechanisms

A question can fail to have an answer in two different ways, and one guard can't catch both.

**Unrelated questions** score low across every chunk — nothing in the corpus is close in meaning. A **similarity threshold** catches these before the model is ever called: if the top score falls below the threshold, the system refuses immediately.

**On-topic questions with no answer in the corpus** score *high* — the retrieved chunks are genuinely relevant, they just don't contain the specific fact being asked about. The threshold can't catch these, because they look identical to a real question by every score-based measure. Only the **prompt's fallback instruction** — telling the model to say "I don't know" when the provided text doesn't answer the question — catches this case, because it requires actually reading the content.

Measuring both categories separately (rather than a single "did it refuse" number) is what revealed this: on a test set, unrelated questions were refused correctly by the threshold every time, while roughly a third of on-topic-but-uncovered questions slipped through the threshold and had to be caught by the prompt instead. Neither mechanism substitutes for the other.

---

## The similarity threshold

`THRESHOLD = 0.43`, set from `.env`.

This was derived from measured score distributions rather than chosen by feel. Across a labelled test set:

- Genuinely answerable questions scored **0.464–0.843**
- Clearly unrelated questions scored **0.361–0.404**

The two ranges don't overlap — there's a gap between 0.404 and 0.464 — and 0.43 sits inside it. An earlier value of 0.5 sat *above* the gap, which meant the lowest-scoring genuine question was being wrongly refused.

Caveat: this is derived from a small labelled set, and the true boundary may sit lower. It should be re-measured as the document corpus grows.

---

## Setup

Requires Python 3.12+, PostgreSQL, and [Ollama](https://ollama.com) running locally.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

psql postgres
CREATE DATABASE ragdb;
\q

cp .env.example .env      # then fill in your values

uv sync
uv run uvicorn main:app --reload
```

The `semantic` table is created automatically at startup.

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## API

**`POST /documents`** — ingest a document, chunked and embedded automatically

```json
{
  "document_name": "cloud.txt",
  "text": "First paragraph.\n\nSecond paragraph."
}
```

Re-ingesting a document replaces its existing chunks rather than duplicating them.

**`GET /search?q=...`** — semantic search, returns the top 3 chunks ranked by similarity score. No answer generation — retrieval only.

**`POST /ask?q=...`** — full pipeline: retrieve, ground, answer. Returns the answer, its similarity score, and the source chunks used.

```json
{
  "query": "Why is cloud security important?",
  "reply": "...",
  "score": 0.76,
  "top3": [ { "text": "...", "score": 0.76, "document_name": "cloud.txt", "chunk_index": 2 } ]
}
```

When the top score falls below the threshold:

```json
{ "query": "How do I bake bread?", "reply": "I don't know", "score": 0.34, "top3": [] }
```

---

## Known limitations

- Embeddings are scored in a Python loop rather than with `pgvector` — doesn't scale past a small corpus
- No transaction around delete-then-reinsert during re-ingestion
- `document_name` is client-supplied, so two different files with the same name will overwrite each other
- No maximum chunk size — a document with one very long paragraph produces one oversized chunk

## Running with Docker

Requires Docker Desktop only. Postgres and Ollama both run in containers.

    git clone https://github.com/sarmadnagori/rag-document-qa
    cd rag-document-qa
    cp .env.example .env
    docker compose up

First run downloads about 2 GB of images. Once the containers are up, pull
both models into the Ollama container:

    docker compose exec ollama ollama pull llama3.2
    docker compose exec ollama ollama pull nomic-embed-text

Another ~2.3 GB, once only — it persists in a named volume.

Then open http://localhost:8002/docs

Ingestion embeds every chunk, and Ollama in a container runs on CPU, so a
long document takes a while. For faster inference, remove the `ollama`
service and set `OLLAMA_HOST=http://host.docker.internal:11434` to use a
local Ollama install.

`docker compose down -v` wipes the database and the downloaded models.