from tools.registry import TOOL_REGISTRY


class ToolWrapper:
    @staticmethod
    def call(name: str, args: dict) -> str:
        # 1. Lookup
        if name not in TOOL_REGISTRY:
            return f"Error: tool '{name}' does not exist."

        tool = TOOL_REGISTRY[name]

        # 2. Validate — Pydantic checks types and constraints
        try:
            params = tool["params_model"](**args)
        except Exception as e:
            return f"Validation error for '{name}': {e}"

        # 3. Execute + 4. Return
        try:
            return str(tool["func"](params))
        except Exception as e:
            return f"Execution error for '{name}': {e}"

    @staticmethod
    def catalog() -> list[dict]:
        return [
            {
                "name": name,
                "description": tool["description"],
                "input_schema": tool["params_model"].model_json_schema(),
            }
            for name, tool in TOOL_REGISTRY.items()
        ]
