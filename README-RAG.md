# Document Analyst with RAG

Extension of the QA agent (see `README.md`) with a document ingestion and retrieval pipeline.
Documents are chunked, embedded, and stored in PostgreSQL + pgvector. The agent queries them
by semantic similarity using the `search_documents_tool`.

## What was added

```
01-ChatBot/
├── pipeline/                    # Lesson 3 — extraction pipeline
│   ├── loaders.py               # load_document(): TXT, PDF, DOCX, CSV
│   ├── splitter.py              # RecursiveCharacterTextSplitter + should_chunk()
│   ├── schemas.py               # RomanianInvoice (Pydantic)
│   └── extraction.py            # ExtractionPipeline — LLM structured extraction
├── rag/                         # Lesson 4 — RAG pipeline
│   ├── database.py              # engine, SessionLocal, Base, transaction()
│   ├── models.py                # Document + DocumentChunk (SQLAlchemy)
│   ├── repositories.py          # DocumentRepository + ChunkRepository
│   ├── embeddings.py            # EmbeddingService (lazy singleton, 384-dim, query cache)
│   └── rag_service.py           # RAGService — ingest, search, get_context
├── alembic/                     # schema migrations
│   └── versions/
│       ├── 193f312f2b07_init.py                       # pgvector extension + tables
│       └── bdd6591e7f9d_add_hnsw_index_on_chunks.py   # HNSW cosine index
├── alembic.ini
├── docker-compose.yml           # pgvector/pgvector:pg16, port 5433
└── sample_docs/                 # drop documents here — auto-ingested at startup (gitignored)
```

New tools registered in `tools/basic_tools.py`:

| Tool                    | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| `extract_invoice_tool`  | Extracts structured data from an invoice (PDF, DOCX, TXT) |
| `search_documents_tool` | Semantic search across all ingested documents             |

## Setup (first time)

> Prerequisite: create a virtualenv and `pip install -r requirements.txt`
> (see [README.md](README.md) for the venv steps).

1. Configure environment — copy the template and fill in real values:

   ```bash
   cp .env.example .env
   ```

   You must set `ANTHROPIC_API_KEY` and the four Postgres variables. The
   `DATABASE_URL` user/password/db must match the `POSTGRES_*` values and use
   port `5433`. See `.env.example` for a worked example.

2. Start the database:

   ```bash
   docker compose up -d
   ```

3. Run migrations (creates the `vector` extension, both tables, and the HNSW index):

   ```bash
   alembic upgrade head
   ```

4. Add documents to `sample_docs/` — any `.txt`, `.pdf`, `.docx`, or `.csv` file.

5. Start the agent (auto-ingests new files in `sample_docs/` on every restart):
   ```bash
   python agent.py
   ```

## Run (day-to-day)

```bash
python agent.py
```

New files dropped into `sample_docs/` are picked up automatically. Already-ingested files are skipped.

## Example session

```
QA Agent ready. Type your question or 'exit' to quit.

You: extract sample_docs/factura_001.txt
Agent: Invoice #1042, issued 2024-03-15, supplier Alfa SRL, total 3.250,00 RON.
       Products: consultanta IT (2h), licenta software.

You: Cand se semneaza contractul de vanzare-cumparare?
Agent: Contractul de vanzare-cumparare se semneaza in termen de 60 de zile
       de la data semnarii antecontractului. [sursa: antecontract.docx | chunk 2]

You: exit
```

## How RAG works

```
User question
      │
      ▼
EmbeddingService.embed(query)          — 384-dim vector via paraphrase-multilingual-MiniLM-L12-v2
      │
      ▼
ChunkRepository.similarity_search()   — cosine distance via HNSW index (<=> operator)
      │
      ▼
RAGService.get_context()              — threshold filter (≥0.4) → dynamic top_k → token budget
      │
      ▼
ToolMessage injected into ReAct loop  — LLM sees context + source citations, answers grounded
```

Three best practices applied in `get_context()`:

- **Context window management** — ~4 chars = 1 token, hard cap at 3000 tokens
- **Dynamic top_k** — if 3+ chunks score > 0.7, use only those 3; otherwise use top 5
- **Source attribution** — every chunk is tagged `[filename | chunk N | score X.XX]`

The `search_documents_tool` also returns a **confidence score** (avg similarity of top
results); below 0.3 it attaches a low-confidence warning. If nothing clears the
threshold it returns a clear "no relevant info" message rather than an error.

**Robustness:**

- **Query embedding cache** — `embed()` is `lru_cache`d, so repeated questions skip re-encoding.
- **Graceful degradation** — if the DB is unreachable the agent still starts (RAG just
  goes offline); a missing `sample_docs/` folder is skipped, not fatal.
- **Duplicate guard** — re-ingesting a file already in the DB (matched by filename) is skipped.

## Schema migrations

Alembic manages the full schema. To apply after a fresh clone:

```bash
alembic upgrade head    # runs all pending migrations in order
```

To generate a migration after changing a model:

```bash
# edit rag/models.py first, then:
alembic revision --autogenerate -m "description"
# review the generated file in alembic/versions/, then:
alembic upgrade head
```

> Note: `CREATE EXTENSION IF NOT EXISTS vector` must always be added manually
> to the first migration's `upgrade()` — autogenerate never detects extensions.
