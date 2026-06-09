import json
from datetime import datetime
from zoneinfo import ZoneInfo

from ddgs import DDGS

from tools.registry import register_tool
from tools.params_models import CalculatorParams, ExtractInvoiceParams, GetDatetimeParams, SearchDocumentsParams, WebSearchParams
from pipeline.extraction import ExtractionPipeline
from pipeline.schemas import RomanianInvoice
from rag.database import transaction
from rag.rag_service import RAGService


@register_tool
def calculator(params: CalculatorParams) -> str:
    """Evaluates a safe mathematical expression and returns the numeric result."""
    return str(eval(params.expression, {"__builtins__": {}}, {}))

@register_tool
def get_datetime(params: GetDatetimeParams) -> str:
    """Returns the current date and time for the given IANA timezone name."""
    return datetime.now(ZoneInfo(params.timezone)).isoformat()

@register_tool
def web_search(params: WebSearchParams) -> str:
    """Searches the web for the given query and returns the top results as text."""
    results = DDGS().text(params.query, max_results=params.max_results)
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)


def extract_invoice_data(file_path: str) -> RomanianInvoice:
    pipeline = ExtractionPipeline()
    result = pipeline.process(file_path, RomanianInvoice)
    return result


# error handling returns dict not raise
# json.dumps used instead of raw dict so ToolWrapper.call()'s str() produces clean JSON for the LLM
@register_tool
def extract_invoice_tool(params: ExtractInvoiceParams) -> str:
    """Extracts structured data from an invoice file (PDF, DOCX, TXT).
    Use this tool when the user asks to extract, parse or process an invoice or financial document.
    Returns: invoice number, date, client, supplier, total amount and list of products."""
    try:
        result = extract_invoice_data(params.file_path)
        return json.dumps(result.model_dump(), ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"success": False, "error": f"File not found: {params.file_path}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# error handling: no context, low confidence
# sources list + confidence score in return
@register_tool
def search_documents_tool(params: SearchDocumentsParams) -> str:
    """Searches all ingested documents using semantic similarity and returns relevant context.
    Use this tool when the user asks a question about the content of loaded documents,
    wants to find information across documents, or asks anything that may be answered
    by stored invoices, contracts, or other files.
    Returns: relevant text excerpts with source citations and a confidence score."""
    try:
        with transaction() as db:
            rag = RAGService(db)
            results = rag.search(params.query, top_k=10)

            # no context: clear message, not an error
            if not results:
                return json.dumps({"success": False, "message": "Nu am găsit informații relevante în documente."})

            context = rag.get_context(params.query, top_k=10)

            # print retrieved chunks to terminal for visibility — agent does not see this
            print(f"\n{'='*60}")
            print(f"  RAG Search: '{params.query}'")
            print(f"{'='*60}")
            for i, (chunk, score) in enumerate(results[:5], 1):
                print(f"\n  [{i}] {chunk.document.filename}  |  chunk {chunk.chunk_index}  |  score {score:.2f}")
                for line in chunk.content[:300].strip().splitlines():
                    print(f"      │  {line}")
                print()
            print(f"{'='*60}\n")

            # source attribution: separate sources list for the agent
            sources = [
                {
                    "filename": chunk.document.filename,
                    "chunk_index": chunk.chunk_index,
                    "score": f"{score:.0%}",
                }
                for chunk, score in results[:5]
            ]

            # confidence = avg similarity of top results
            confidence = sum(score for _, score in results[:5]) / len(results[:5])

            result = {"success": True, "context": context, "sources": sources, "confidence": round(confidence, 2)}

            # low confidence: include answer but warn the user
            if confidence < 0.3:
                result["warning"] = f"Low confidence ({confidence:.0%}) — results may not be relevant."

            return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
