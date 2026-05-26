from datetime import datetime
from zoneinfo import ZoneInfo

from ddgs import DDGS

from tools.registry import register_tool
from tools.params_models import CalculatorParams, GetDatetimeParams, WebSearchParams


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
