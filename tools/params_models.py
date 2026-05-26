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
