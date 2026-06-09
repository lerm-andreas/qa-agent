from pydantic import BaseModel, Field

class CalculatorParams(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate, e.g. '2 + 3 * 4'",
        min_length=1,
    )

class GetDatetimeParams(BaseModel):
    timezone: str = Field(
        default="UTC",
        description="IANA timezone name, e.g. 'Europe/Bucharest'",
    )

class WebSearchParams(BaseModel):
    query: str = Field(
        description="Search terms to look up on the web",
        min_length=2,
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20,
    )

class ExtractInvoiceParams(BaseModel):
    file_path: str = Field(
        description="Relative path to the invoice file (e.g. sample_docs/factura_001.txt)",
        min_length=1,
    )

class SearchDocumentsParams(BaseModel):
    query: str = Field(
        description="Question or topic to search for across all ingested documents",
        min_length=2,
    )
