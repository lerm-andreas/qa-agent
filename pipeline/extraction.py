import json
from pathlib import Path
from typing import Type

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from pipeline.loaders import load_document
from pipeline.splitter import should_chunk, split

load_dotenv()

_EXTRACTION_SYSTEM_PROMPT = """Ești un expert în extragerea datelor din documente românești.
Extrage cu atenție toate câmpurile cerute.
Dacă un câmp lipsește, lasă-l null."""


class ExtractionPipeline:

    def __init__(self, provider: str = "anthropic"):
        self.llm = self._create_llm(provider)

    def _create_llm(self, provider: str):
        providers = {
            "anthropic": lambda: ChatAnthropic(model="claude-haiku-4-5-20251001"),
            "gemini":    lambda: ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
            "ollama":    lambda: ChatOllama(model="llama3.2"),
        }
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        return providers[provider]()


    def process(self, file_path: str, schema: Type[BaseModel]) -> BaseModel:
        docs = load_document(file_path)

        if should_chunk(docs):
            chunks = split(docs)
            # "info cheie e la început", combine first 3 chunks only
            text = "\n\n".join(c.page_content for c in chunks[:3])
        else:
            text = "\n\n".join(doc.page_content for doc in docs)

        messages = [
            SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"Extrage datele din acest document:\n\n{text}"),
        ]
        return self.llm.with_structured_output(schema).invoke(messages)

    # save_json(data, doc_type) with mkdir + model_dump + ensure_ascii=False
    # getattr fallback handles schemas that don't have a numar field
    def save_json(self, result: BaseModel, doc_type: str = "factura") -> Path:
        out = Path("extracted_data") / doc_type
        out.mkdir(parents=True, exist_ok=True)
        filename = getattr(result, "numar", "document").replace("/", "_") + ".json"
        output_path = out / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        return output_path
