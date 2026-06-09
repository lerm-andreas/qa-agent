from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,          # 1 Document per row
    Docx2txtLoader,     # 1 Document per file, plain text only
    PyPDFLoader,        # 1 Document per page, metadata includes page number
    TextLoader,         # 1 Document per file
)
from langchain_core.documents import Document

# registry pattern: extension → loader factory
# TextLoader uses lambda (not bare class) to bake in encoding="utf-8"
# for Romanian diacritics
LOADER_REGISTRY = {
    "txt":  lambda path: TextLoader(path, encoding="utf-8"),
    "pdf":  PyPDFLoader,
    "docx": Docx2txtLoader,
    "csv":  CSVLoader,
}


def load_document(path: str) -> List[Document]:
    file_ext = Path(path).suffix.lstrip(".").lower()
    if file_ext not in LOADER_REGISTRY:
        raise ValueError(f"Tip nesuportat: .{file_ext}")
    return LOADER_REGISTRY[file_ext](path).load()
