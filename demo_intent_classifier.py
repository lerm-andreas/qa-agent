"""Compare TF-IDF+LR intent classifier against an LLM baseline

Run AFTER training the artifact:
    python train_intent.py           # builds ml/intent_classifier.joblib (once)
    python demo_intent_classifier.py

Requires ANTHROPIC_API_KEY in .env for the LLM baseline section.
"""

import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.metrics import classification_report

from ml.intent_classifier import DEFAULT_MODEL_PATH, IntentClassifier
from ml.intent_data import LABELS, TEST_DATA

load_dotenv()

# ── Haiku 4.5 pricing
_HAIKU_INPUT_RATE_PER_M  = 1.00
_HAIKU_OUTPUT_RATE_PER_M = 5.00


# ── LLM BASELINE
# temperature=0 makes the output deterministic — for a one-word classification
# task there should be no creative variation; we want reproducible results.
_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
_SYSTEM = (
    "You are an intent classifier. Given a user query, reply with exactly one word: "
    "'search', 'extract', or 'summarize'. No punctuation, no explanation."
)


def detect_intent_llm(query: str) -> tuple[str, int, int]:
    """Send query to Haiku, return (intent_label, input_tokens, output_tokens).

    Returns token counts so the caller can accumulate them across calls and
    price the full run rather than estimating from a fixed $/call figure.
    """
    response = _llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=query),
    ])
    intent = response.content.strip().lower()

    # The prompt says "one word", but LLMs occasionally add punctuation.
    # Scan for any of our labels so a response like "search." still maps correctly.
    for label in ("search", "extract", "summarize"):
        if label in intent:
            intent = label
            break

    meta    = response.usage_metadata or {}
    in_tok  = meta.get("input_tokens",  0)
    out_tok = meta.get("output_tokens", 0)
    return intent, in_tok, out_tok


# ── BENCHMARK HARNESS ─────────────────────────────────────────────────────────
def benchmark(func, *args, iterations=100):
    """Măsoară latența medie."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return {
        "avg_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def main() -> None:
    classifier = IntentClassifier(model_path=DEFAULT_MODEL_PATH)

    queries = [q for q, _ in TEST_DATA]
    gold    = [g for _, g in TEST_DATA]

    # ── 1. ACCURACY PASS
    print(f"\nAccuracy pass — {len(TEST_DATA)} test queries …\n")

    sklearn_preds = []
    llm_preds     = []
    total_in_tok  = 0
    total_out_tok = 0

    for query in queries:
        sklearn_preds.append(classifier.predict(query)[0])
        intent, in_tok, out_tok = detect_intent_llm(query)
        llm_preds.append(intent)
        total_in_tok  += in_tok
        total_out_tok += out_tok
        print(f"  [{intent:>9}]  {query[:55]}")

    sklearn_acc = sum(p == g for p, g in zip(sklearn_preds, gold)) / len(gold)
    llm_acc     = sum(p == g for p, g in zip(llm_preds,     gold)) / len(gold)

    # ── 2. LATENCY
    sample_query = queries[0]
    llm_stats = benchmark(detect_intent_llm,   sample_query, iterations=20)
    ml_stats  = benchmark(classifier.predict,  sample_query, iterations=100)

    print(f"\nLLM: {llm_stats['avg_ms']:.0f}ms")
    print(f"ML:  {ml_stats['avg_ms']:.1f}ms")

    # ── 3. COST ESTIMATE
    # Real usage_metadata averages from the accuracy pass, scaled to 1 000 calls.
    n_calls = len(TEST_DATA)
    avg_in  = total_in_tok  / n_calls
    avg_out = total_out_tok / n_calls

    llm_cost_per_1k = (
        avg_in  * _HAIKU_INPUT_RATE_PER_M  / 1_000_000 +
        avg_out * _HAIKU_OUTPUT_RATE_PER_M / 1_000_000
    ) * 1_000

    # ── 4. COMPARISON TABLE
    print("\n" + "=" * 64)
    print(f"{'Metric':<24} {'sklearn (TF-IDF+LR)':>18} {'LLM (Haiku 4.5)':>18}")
    print("-" * 64)
    print(f"{'Accuracy':<24} {sklearn_acc:>17.1%} {llm_acc:>17.1%}")
    print(f"{'Avg latency (ms)':<24} {ml_stats['avg_ms']:>17.1f} {llm_stats['avg_ms']:>17.1f}")
    print(f"{'  min / max (ms)':<24} {ml_stats['min_ms']:>7.1f} / {ml_stats['max_ms']:<8.1f} "
          f"{llm_stats['min_ms']:>7.1f} / {llm_stats['max_ms']:<8.1f}")
    print(f"{'Cost / 1 000 calls (USD)':<24} {'0.0000':>18} {llm_cost_per_1k:>17.4f}")
    print(f"{'  avg tokens (in/out)':<24} {'n/a':>18} {avg_in:>7.0f} / {avg_out:<7.0f}")
    print(f"{'Works offline':<24} {'Yes':>18} {'No':>18}")
    print("=" * 64)

    speedup   = llm_stats['avg_ms'] / ml_stats['avg_ms'] if ml_stats['avg_ms'] > 0 else float("inf")
    acc_diff  = abs(sklearn_acc - llm_acc) * 100
    print(
        f"\nTrade-off: sklearn is ~{speedup:.0f}× faster and saves ${llm_cost_per_1k:.4f} "
        f"per 1 000 calls with {acc_diff:.1f}% accuracy difference on this test set."
    )

    # ── 5. PER-CLASS BREAKDOWN
    # classification_report gives precision/recall/F1 per class, not just aggregate
    # accuracy — useful for spotting which intent is hardest to classify.
    print("\n── sklearn classification_report ──────────────────────────────")
    print(classification_report(gold, sklearn_preds, target_names=LABELS))

    print("── LLM classification_report ──────────────────────────────────")
    print(classification_report(gold, llm_preds, target_names=LABELS))


if __name__ == "__main__":
    main()
