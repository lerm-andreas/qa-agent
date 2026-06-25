import time

from agent import QAAgent

QUESTIONS = [
    "Care este penalizarea de întârziere prevăzută în documente?",
    "Ce modalitate de plată este menționată în contracte?",
    "Care este durata contractului de consultanță?",
    "Există clauze de confidențialitate în documente?",
    "Cum se poate rezilia contractul?",
]


def main():
    agent = QAAgent(session_id="cache_demo")

    print("=" * 62)
    print("PROMPT CACHING DEMO — QAAgent")
    print("Model: claude-haiku-4-5-20251001")
    print("=" * 62)
    print("(primul apel = cache MISS, restul = cache HIT)\n")

    latencies = []
    for i, question in enumerate(QUESTIONS, start=1):
        t0 = time.perf_counter()
        answer = agent.chat(question)
        lat = time.perf_counter() - t0
        latencies.append(lat)
        print(f"→ Apel {i}: {question}")
        print(f"   latency={lat:.2f}s  |  răspuns: {answer[:100].strip()}...\n")

    print("── Sumar economii (cumulat) ────────────────────────────────")
    avg_miss = latencies[0]
    avg_hit  = sum(latencies[1:]) / len(latencies[1:])
    print(f"   latență medie MISS={avg_miss:.2f}s  HIT={avg_hit:.2f}s  delta={avg_miss - avg_hit:+.2f}s")
    for k, v in agent.cache_metrics.report().items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
