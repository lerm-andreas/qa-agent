from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from prompts.registry import get_prompt_registry
from rag.database import transaction
from rag.rag_service import RAGService
from tools import ToolWrapper

load_dotenv()


class QAAgent:
    def __init__(self, provider: str = "anthropic", max_iterations: int = 15):
        self.llm = self._create_llm(provider)
        self.max_iterations = max_iterations
        self.history: list = []
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

    def chat(self, message: str) -> str:
        messages = [SystemMessage(self.system_prompt), *self.history, HumanMessage(message)]
        answer = self._react_loop(messages)

        # comment next 2 lines when you don't want history (saves tokens)
        self.history.append(HumanMessage(message))
        self.history.append(AIMessage(answer))

        return answer

    def _react_loop(self, messages: list) -> str:
        llm_with_tools = self.llm.bind_tools(ToolWrapper.catalog())

        for _ in range(self.max_iterations):
            response = llm_with_tools.invoke(messages)  # THINK
            messages.append(response)

            if not response.tool_calls:                  # no tools → final answer
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
