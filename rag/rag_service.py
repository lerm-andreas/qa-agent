from pathlib import Path

from sqlalchemy.orm import Session

from pipeline.loaders import load_document
from pipeline.splitter import split
from rag.embeddings import EmbeddingService
from rag.models import DocumentChunk
from rag.repositories import DocumentRepository, ChunkRepository

DEFAULT_THRESHOLD = 0.4
MAX_TOKENS = 3000

class RAGService:

    def __init__(self, db: Session) -> None:
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self._embeddings = EmbeddingService()

    def ingest(self, file_path: str) -> None:
        path = Path(file_path)

        # skip if already ingested — avoid duplicate documents
        if self.doc_repo.get_by_filename(path.name):
            print(f"Already ingested: {path.name} — skipping")
            return

        docs = load_document(file_path)

        # always split for storage — small docs become a single chunk naturally,
        # multi-page PDFs (N Document objects) get merged and re-split correctly
        chunks = split(docs)

        full_content = "\n\n".join(doc.page_content for doc in docs)
        doc = self.doc_repo.create(
            filename=path.name,
            content=full_content,
            metadata={"source": file_path},
        )

        # embed all chunks in one batch call — faster than N individual embed() calls
        texts = [c.page_content for c in chunks]
        embeddings = self._embeddings.embed_batch(texts)

        self.chunk_repo.create_chunks_batch([
            {
                "document_id": doc.id,
                "content": text,
                "chunk_index": i,
                "embedding": emb,
            }
            for i, (text, emb) in enumerate(zip(texts, embeddings))
        ])

        print(f"Ingested: {path.name} ({len(chunks)} chunks)")

    def ingest_folder(self, folder: str = "sample_docs") -> None:
        folder_path = Path(folder)
        # folder may not exist on a fresh clone (sample_docs/ is gitignored) —
        # skip quietly instead of crashing the agent at startup
        if not folder_path.is_dir():
            print(f"⚠ Folder not found: {folder} — nothing to ingest")
            return
        supported = {".txt", ".pdf", ".docx", ".csv"}
        files = sorted(f for f in folder_path.iterdir() if f.suffix.lower() in supported)
        for file in files:
            self.ingest(str(file))

    # encode query → similarity_search → (chunk, score) list
    # top_k=10: wide net — get_context will trim down via dynamic top_k
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        query_embedding = self._embeddings.embed(query)
        return self.chunk_repo.similarity_search(query_embedding, top_k=top_k)

    def get_context(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = DEFAULT_THRESHOLD,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        results = self.search(query, top_k=top_k)

        # filter out chunks below similarity threshold
        relevant = [(chunk, score) for chunk, score in results if score >= threshold]
        if not relevant:
            return ""

        # dynamic top_k: if first 3 are high quality, use only those
        # prevents context dilution with weaker results
        high_quality = [(c, s) for c, s in relevant if s > 0.7]
        candidates = high_quality[:3] if len(high_quality) >= 3 else relevant[:5]

        # context window management: ~4 chars = 1 token
        parts = []
        total_tokens = 0
        for chunk, score in candidates:
            chunk_tokens = len(chunk.content) // 4
            if total_tokens + chunk_tokens > max_tokens:
                break
            # inline citation format for LLM
            parts.append(
                f"[{chunk.document.filename} | chunk {chunk.chunk_index} | score {score:.2f}]\n"
                f"{chunk.content}"
            )
            total_tokens += chunk_tokens

        return "\n\n".join(parts)
