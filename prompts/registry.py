from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    prompt: str
    description: str = ""


class PromptRegistry:
    def __init__(self, folder: str):
        self._templates = self._load(folder)

    def _load(self, folder: str) -> dict[str, PromptTemplate]:
        templates = {}
        for path in Path(folder).rglob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            tpl = PromptTemplate(**data)
            templates[tpl.name] = tpl
        return templates

    def render(self, name: str, **variables) -> str:
        template = self._templates[name]
        return Template(template.prompt).render(**variables)


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry(folder="prompts/")
