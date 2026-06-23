# Agent QA cu Tools + Prompts

A ReAct-pattern QA agent that uses tools and externalized prompts to answer questions accurately.

> **RAG / document analysis (HW4):** this project was extended with a document
> extraction + RAG pipeline. See **[README-RAG.md](README-RAG.md)** for that part —
> the two READMEs will be merged later.

## Project structure

```
01-ChatBot/
├── agent.py                 # QA agent / LLM orchestration (entry point)
├── demo_prompt_cache.py     # prompt caching demo — latency + token savings
├── tools/
│   ├── __init__.py          # exports ToolWrapper
│   ├── registry.py          # TOOL_REGISTRY + @register_tool
│   ├── params_models.py     # Pydantic BaseModel per tool
│   ├── basic_tools.py       # calculator, get_datetime, web_search
│   └── tool_wrapper.py      # ToolWrapper.call() + ToolWrapper.catalog()
├── prompts/
│   ├── registry.py          # PromptRegistry (load YAML + Jinja2)
│   ├── planner.yaml         # main system prompt for the ReAct agent
│   ├── analyst.yaml         # prompt for analytical tasks
│   ├── summary.yaml         # prompt for summarization tasks
│   ├── extract.yaml         # prompt for data extraction tasks
│   └── fixed_context.txt    # static domain reference block (cached prefix)
├── rag/
│   ├── memory.py            # PersistentMemory — load/save conversation history
│   └── cache_metrics.py     # CacheMetrics — token savings accumulator
├── alembic/versions/
│   └── *_add_chat_messages_table.py  # migration: chat_messages table
├── requirements.txt
└── .env                     # API keys (never commit this)
```

## Setup

1. Create and activate the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API key to `.env`:
   ```
   GOOGLE_API_KEY=your-key-here
   # or
   ANTHROPIC_API_KEY=your-key-here
   ```

## Run

```bash
.venv/bin/python agent.py
```

The agent starts an interactive chat session in the terminal:

```
QA Agent ready. Type your question or 'exit' to quit.

You: What is 137 * 42?
Agent: 137 * 42 = 5754

You: What time is it in Bucharest?
Agent: The current time in Bucharest is 14:32:10 on 2026-05-25.

You: exit
```

Conversation history is persisted to PostgreSQL and survives restarts — each `session_id` keeps its own history window. Type `exit` to quit.

## Testing tools without the LLM

To test individual tools without making any API calls:

```bash
.venv/bin/python -c "
from tools import ToolWrapper
print(ToolWrapper.call('calculator', {'expression': '137 * 42'}))
print(ToolWrapper.call('get_datetime', {'timezone': 'Europe/Bucharest'}))
print(ToolWrapper.call('web_search', {'query': 'Python news', 'max_results': 2}))
"
```

## Prompt caching

The agent caches its static prefix (tool catalog + system prompt + domain reference block) using Anthropic's prompt caching. On repeated calls the cached portion is read at ~0.10× the normal input cost. Run the demo to see token savings and latency improvement:

```bash
.venv/bin/python demo_prompt_cache.py
```

## Tools

| Tool | Description |
|------|-------------|
| `calculator` | Evaluates a mathematical expression and returns the result |
| `get_datetime` | Returns the current date and time for a given IANA timezone |
| `web_search` | Searches the web using DuckDuckGo and returns the top results |

## Prompts

| File | Purpose |
|------|---------|
| `planner.yaml` | Main system prompt — drives the ReAct loop |
| `analyst.yaml` | Structured analysis and comparison tasks |
| `summary.yaml` | Condensing long text into key points |
| `extract.yaml` | Pulling structured data out of unstructured text |

## How it works

The agent follows the **ReAct pattern** (Reason + Act):

1. **Think** — the LLM receives the user message + system prompt + available tools
2. **Act** — if the LLM needs data, it calls one or more tools in the same turn
3. **Observe** — tool results are fed back to the LLM
4. **Repeat** — until the LLM produces a final answer (or `max_iterations` is reached)
