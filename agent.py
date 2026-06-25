from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from ml.intent_classifier import DEFAULT_MODEL_PATH, IntentClassifier
from prompts.registry import get_prompt_registry
from rag.cache_metrics import CacheMetrics
from rag.database import transaction
from rag.memory import PersistentMemory
from rag.rag_service import RAGService
from tools import ToolWrapper

load_dotenv()

# Romanian document-domain reference block
# Loaded from prompts/fixed_context.txt to keep agent.py free of large static content.
# Two goals: (1) useful extraction guidance; (2) pushes the cached prefix over Haiku's
# token floor so cache_control actually fires. Must stay IDENTICAL across requests.
FIXED_CONTEXT = (Path(__file__).parent / "prompts" / "fixed_context.txt").read_text(encoding="utf-8")


class QAAgent:
    def __init__(self, provider: str = "anthropic", session_id: str = "default", max_iterations: int = 15):
        self.llm = self._create_llm(provider)
        self.max_iterations = max_iterations
        # conversation history now lives in PostgreSQL (survives restarts),
        # keyed by session_id — replaces the old in-memory self.history list
        self.session_id = session_id
        self.memory = PersistentMemory(window=10)
        self.cache_metrics = CacheMetrics()   # accumulates prompt-cache token counts
        # Load trained intent classifier once at startup (model loads
        # once, ~50 ms cold-start paid here rather than per predict call).
        # Graceful degradation: agent still works if train_intent.py hasn't been run.
        try:
            self.classifier: IntentClassifier | None = IntentClassifier(DEFAULT_MODEL_PATH)
        except Exception:
            self.classifier = None
        self.system_prompt = get_prompt_registry().render(
            "planner", role="QA expert", max_words=50
        )
        # ingest all files from sample_docs/ at startup — new files are picked up
        # on each restart; already-ingested files are silently skipped
        # graceful degradation: agent still works if DB is unavailable
        try:
            with transaction() as db:
                RAGService(db).ingest_folder()
        except Exception as e:
            print(f"⚠ RAG ingestion skipped — DB unavailable: {e}")

    def _create_llm(self, provider: str):
        providers = {
            "anthropic": lambda: ChatAnthropic(model="claude-haiku-4-5-20251001"),
            "gemini":    lambda: ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
            "ollama":    lambda: ChatOllama(model="llama3.2"),
        }
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}")
        return providers[provider]()

    # load → invoke → save
    def chat(self, message: str) -> str:
        # Classify intent before the LLM call (predict + confidence).
        # Prepend as a bracket annotation so the LLM can frame its response accordingly
        # (e.g. "search" → locate/list, "extract" → pull specific field, "summarize" → condense).
        # We annotate the LLM message but save the original text to memory so history
        # stays human-readable.
        if self.classifier is not None:
            intent, conf = self.classifier.predict(message)
            print(f"[intent] {intent} (conf={conf:.2f})")
            # Only inject when confident
            # Here the fallback is skipping the annotation so the LLM reasons from
            # the raw message rather than a potentially wrong hint.
            if conf >= 0.7:
                user_message_with_intent = f"[intent: {intent}] {message}"
            else:
                user_message_with_intent = message
        else:
            user_message_with_intent = message

        # 1. LOAD — last N turns from Postgres, mapped back to LangChain messages.
        history = [
            HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in self.memory.load_messages(self.session_id)
        ]

        # 2. INVOKE — system prompt + history + new turn through the ReAct loop.
        # Build the system message as a list of blocks
        # cache_control on the LAST block sets the cut-line: everything before messages
        # (tools + system prompt + FIXED_CONTEXT) becomes the cached prefix.
        system_msg = SystemMessage(content=[
            {"type": "text", "text": self.system_prompt},
            {"type": "text", "text": FIXED_CONTEXT,
             "cache_control": {"type": "ephemeral"}},  # cut-line after this block
        ])
        messages = [system_msg, *history, HumanMessage(user_message_with_intent)]
        answer = self._react_loop(messages)

        # 3. SAVE — persist only the user message + final answer, not the tool
        self.memory.save_message(self.session_id, "user", message)
        self.memory.save_message(self.session_id, "assistant", answer)

        return answer

    def _react_loop(self, messages: list) -> str:
        llm_with_tools = self.llm.bind_tools(ToolWrapper.catalog())

        for _ in range(self.max_iterations):
            response = llm_with_tools.invoke(messages)  # THINK
            messages.append(response)

            if not response.tool_calls:                  # no tools → final answer
                # record prompt-cache token counts for this turn
                # langchain-anthropic now uses "ephemeral_5m_input_tokens" for cache writes;
                details  = (response.usage_metadata or {}).get("input_token_details", {})
                creation = details.get("ephemeral_5m_input_tokens", 0) or details.get("cache_creation", 0)
                read     = details.get("cache_read", 0)
                total_in = (response.usage_metadata or {}).get("input_tokens", 0)
                fresh    = max(total_in - creation - read, 0)
                self.cache_metrics.record(creation, read, fresh)
                print(f"[cache] creation={creation} read={read} fresh={fresh} "
                      f"hit_rate={self.cache_metrics.hit_rate*100:.0f}%")

                content = response.content
                if isinstance(content, list):            # Gemini returns content blocks
                    return " ".join(b["text"] for b in content if isinstance(b, dict) and "text" in b)
                return content

            for tc in response.tool_calls:               # ACT — multiple calls per turn
                result = ToolWrapper.call(tc["name"], tc["args"])
                messages.append(                         # OBSERVE
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        raise RuntimeError("Max iterations reached without a final answer")


if __name__ == "__main__":
    agent = QAAgent(provider="anthropic")

    print("QA Agent ready. Type your question or 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        print(f"Agent: {agent.chat(question)}\n")
