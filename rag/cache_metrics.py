# We track prompt-cache token counts (creation / read / fresh) because Anthropic
# returns exact numbers per call, which lets us compute real savings rather than estimates.
from dataclasses import dataclass


@dataclass
class CacheMetrics:
    # accumulated token counts across all LLM calls in this session
    cache_creation_tokens: int = 0   # tokens written to cache (billed at 1.25×)
    cache_read_tokens: int = 0       # tokens read from cache (billed at 0.10×)
    fresh_input_tokens: int = 0      # uncached input tokens (billed at 1.00×)
    turns: int = 0                   # total LLM invocations recorded
    cache_hits: int = 0              # invocations where cache_read > 0

    # Haiku 4.5 USD per 1M tokens — anthropic.com/pricing
    # cache write = 1.25× input (5-min ephemeral TTL); cache read = 0.10× input
    _INPUT_RATE:   float = 1.00
    _READ_RATE:    float = 0.10

    def record(self, creation: int, read: int, fresh: int) -> None:
        self.cache_creation_tokens += creation
        self.cache_read_tokens += read
        self.fresh_input_tokens += fresh
        self.turns += 1
        if read > 0:
            self.cache_hits += 1

    @property
    def hit_rate(self) -> float:
        # fraction of invocations that read from the cache
        return self.cache_hits / self.turns if self.turns else 0.0

    @property
    def estimated_savings_usd(self) -> float:
        # actual cost of cached reads vs. what they would cost at full input rate
        full_price  = self.cache_read_tokens * self._INPUT_RATE / 1_000_000
        cache_price = self.cache_read_tokens * self._READ_RATE  / 1_000_000
        return full_price - cache_price

    def report(self) -> dict:
        return {
            "turns":                  self.turns,
            "hit_rate":               f"{self.hit_rate * 100:.1f}%",
            "cache_creation_tokens":  self.cache_creation_tokens,
            "cache_read_tokens":      self.cache_read_tokens,
            "fresh_input_tokens":     self.fresh_input_tokens,
            "estimated_savings_usd":  f"${self.estimated_savings_usd:.4f}",
        }
