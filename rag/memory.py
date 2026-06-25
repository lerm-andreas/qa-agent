from rag.database import transaction
from rag.repositories import ChatMessageRepository


class PersistentMemory:

    def __init__(self, window: int = 10) -> None:
        self.window = window   # how many recent turns to load (sliding window)

    # load the last N turns as plain {role, content} dicts
    def load_messages(self, session_id: str) -> list[dict]:
        with transaction() as db:
            rows = ChatMessageRepository(db).latest(session_id, self.window)
            return [{"role": m.role, "content": m.content} for m in rows]

    # persist one turn atomically; transaction() commits on success
    def save_message(self, session_id: str, role: str, content: str) -> None:
        with transaction() as db:
            ChatMessageRepository(db).add(session_id, role, content)
