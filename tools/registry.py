import inspect

from pydantic import BaseModel

TOOL_REGISTRY: dict[str, dict] = {}

def register_tool(func):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Validation 1: exactly one parameter of type BaseModel
    if len(params) != 1 or not issubclass(params[0].annotation, BaseModel):
        raise TypeError(
            f"{func.__name__}: single BaseModel parameter required"
        )

    # Validation 2: docstring required and >= 15 chars (becomes LLM description)
    docstring = (func.__doc__ or "").strip()
    if not docstring:
        raise ValueError(
            f"{func.__name__}: docstring required — it becomes the tool description for the LLM"
        )
    if len(docstring) < 15:
        raise ValueError(
            f"{func.__name__}: docstring too short ({len(docstring)} chars) — LLM needs at least 15"
        )

    TOOL_REGISTRY[func.__name__] = {
        "func": func,
        "params_model": params[0].annotation,
        "description": docstring,
    }
    return func
