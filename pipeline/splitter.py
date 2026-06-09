from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# separators: paragrafe → linii → cuvinte → caractere (cel mai natural la cel mai mecanic)
def split(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# skip chunking if total content fits comfortably in context window
def should_chunk(docs: List[Document], max_size: int = 4000) -> bool:
    total = sum(len(doc.page_content) for doc in docs)
    return total > max_size
