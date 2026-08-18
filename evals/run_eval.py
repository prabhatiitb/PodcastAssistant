import json
import chromadb
from src.qa import answer_question


def evaluate(collection, test_set_path: str, top_k: int = 4):
    with open(test_set_path) as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        result = answer_question(collection, case["question"], top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]

        # --- Answer correctness check (simple substring match) ---
        expected = case.get("expected_answer_contains")
        answer_correct = None
        if expected:
            answer_correct = expected.lower() in answer.lower()

        # --- Retrieval check: did any retrieved chunk fall in the expected timestamp range? ---
        retrieval_hit = None
        expected_range = case.get("expected_timestamp_range")
        if expected_range:
            lo, hi = expected_range
            retrieval_hit = any(lo <= src["start"] <= hi for src in sources)

        results.append({
            "question": case["question"],
            "answer": answer,
            "answer_correct": answer_correct,
            "retrieval_hit": retrieval_hit,
        })

    return results


def print_report(results):
    total = len(results)
    answer_scored = [r for r in results if r["answer_correct"] is not None]
    retrieval_scored = [r for r in results if r["retrieval_hit"] is not None]

    answer_acc = sum(r["answer_correct"] for r in answer_scored) / len(answer_scored) if answer_scored else 0
    retrieval_acc = sum(r["retrieval_hit"] for r in retrieval_scored) / len(retrieval_scored) if retrieval_scored else 0

    print(f"\n=== Evaluation Report ({total} questions) ===")
    print(f"Answer accuracy:    {answer_acc:.0%}")
    print(f"Retrieval hit rate: {retrieval_acc:.0%}\n")

    for r in results:
        status = "PASS" if r["answer_correct"] else "FAIL" if r["answer_correct"] is False else "-"
        print(f"[{status}] Q: {r['question']}")
        print(f"    A: {r['answer'][:150]}...")
        print()


if __name__ == "__main__":
    client_db = chromadb.PersistentClient(path="data/chroma_db")
    collection = client_db.get_collection(name="episode_<your_episode_id>")  # update this

    results = evaluate(collection, "evals/test_questions.json")
    print_report(results)
