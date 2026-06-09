from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey,
    UniqueConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from rag.database import Base



class Document(Base):
    __tablename__ = "documents"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    filename     = Column(String(255), nullable=False, index=True)
    content      = Column(Text, nullable=False)
    doc_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    created_at   = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # back-reference to chunks
    # cascade="all, delete-orphan": deleting a Document deletes all its chunks
    chunks = relationship("DocumentChunk", back_populates="document",
                          cascade="all, delete-orphan")


# one Document → many DocumentChunks (one-to-many)
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content     = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding   = Column(Vector(384), nullable=False)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        # no duplicate chunk positions per document
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_idx"),
        # speeds up "all chunks for document X" queries
        Index("ix_chunks_document_id", "document_id"),
    )
