from functools import lru_cache

from sentence_transformers import SentenceTransformer

# multilingual model recommended for Romanian text, 384-dim output
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingService:
    # class-level variable so the model is loaded
    # once per process, not once per EmbeddingService() call
    _model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        # only download/load on first use
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(MODEL_NAME)
        return EmbeddingService._model

    def embed(self, text: str) -> list[float]:
        # used at query time: embed the user's question.
        # cached so repeated queries skip re-encoding (lesson4.pdf — embedding cache).
        # list(...) returns a fresh copy each call so callers can't mutate the cached value.
        return list(_embed_cached(self.model, text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # used at ingestion time: embed all chunks at once
        # batch encode is significantly faster than N individual embed() calls
        # not cached — ingestion chunks are unique, caching would only waste memory
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


# module-level so the cache survives across short-lived EmbeddingService instances
# (a new one is created per request). model is the process-wide singleton, so the
# key is effectively just `text`. Returns a tuple — hashable + safe to cache.
@lru_cache(maxsize=1000)
def _embed_cached(model: SentenceTransformer, text: str) -> tuple[float, ...]:
    return tuple(model.encode(text, convert_to_numpy=True).tolist())
