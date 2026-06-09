from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from rag.models import Document, DocumentChunk


class DocumentRepository:
    """single API for all Document DB operations."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # flush() writes within transaction but doesn't commit
    # commit() is handled by transaction() context manager, not here
    def create(self, filename: str, content: str, metadata: dict) -> Document:
        doc = Document(filename=filename, content=content, doc_metadata=metadata)
        self.db.add(doc)
        self.db.flush()  # makes doc.id available for chunk FK references
        return doc

    def get_by_id(self, doc_id: int) -> Document | None:
        return self.db.get(Document, doc_id)

    def get_by_filename(self, filename: str) -> Document | None:
        return self.db.query(Document).filter(Document.filename == filename).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Document]:
        return (
            self.db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def delete(self, doc_id: int) -> bool:
        doc = self.db.get(Document, doc_id)
        if doc is None:
            return False
        self.db.delete(doc)
        self.db.flush()
        return True


class ChunkRepository:
    """single API for all DocumentChunk DB operations."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # batch INSERT is much faster than N individual inserts for large documents
    def create_chunks_batch(self, chunks: list[dict]) -> list[DocumentChunk]:
        objects = [DocumentChunk(**chunk) for chunk in chunks]
        self.db.add_all(objects)
        self.db.flush()
        return objects

    def get_document_chunks(self, document_id: int) -> list[DocumentChunk]:
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        # 1 - cosine_distance flips the scale: score 1.0 = identical, 0.0 = unrelated
        # (cosine_distance alone: 0.0 = identical, which is less intuitive)
        similarity = (
            1 - DocumentChunk.embedding.cosine_distance(query_embedding)
        ).label("score")

        # stmt = SQL statement (not yet executed — just a query object)
        stmt = (
            # select both the ORM object and the computed score so each row unpacks as (chunk, score)
            select(DocumentChunk, similarity)
            # eager-load the parent Document in the same query to avoid N+1
            # (needed later when get_context() accesses chunk.document.filename)
            .options(joinedload(DocumentChunk.document))
            # sort by raw distance ascending (smallest distance = most similar = first)
            # using distance directly (not 1-distance) so PostgreSQL can use the HNSW vector index
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        rows = self.db.execute(stmt).all()
        # unpack each Row into a clean Python tuple (DocumentChunk, float)
        return [(chunk, float(score)) for chunk, score in rows]
