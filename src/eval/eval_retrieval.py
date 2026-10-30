import json
import sys
from pathlib import Path
from utils import LatencyTracker, report, save_result

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "pipeline"))

import retrieve
from config import GOLD_RAG_PATH, RAG_TOP_K


def is_relevant(chunk, relevant_pages):
    return any(
        chunk["page_start"] <= page <= chunk["page_end"] for page in relevant_pages
    )


def run_eval(k=RAG_TOP_K):
    gold = json.loads(GOLD_RAG_PATH.read_text(encoding="utf-8"))
    model, retriever, by_id = retrieve.load()

    recalls = []
    reciprocal_ranks = []
    questions = []
    trackers = []

    for item in gold:
        if "relevant_pages" not in item:
            continue

        question = item["question"]
        relevant_pages = item["relevant_pages"]

        tracker = LatencyTracker()
        with tracker.measure("total"), tracker.measure("retrieval"):
            results = retrieve.search(model, retriever, by_id, question, k=k)
        trackers.append(tracker)

        hit_ranks = [
            rank
            for rank, chunk in enumerate(results, start=1)
            if is_relevant(chunk, relevant_pages)
        ]

        recalls.append(1.0 if hit_ranks else 0.0)
        reciprocal_ranks.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)
        questions.append(
            {
                "question": question,
                "relevant_pages": relevant_pages,
                "retrieved_pages": [[c["page_start"], c["page_end"]] for c in results],
                "hit_ranks": hit_ranks,
                "latency": tracker.as_dict(),
            }
        )

        print(f"{question}")
        print(f"   hit at rank(s): {hit_ranks or 'none'}")

    metrics = {}
    if recalls:
        metrics = {
            "k": k,
            "recall_at_k": sum(recalls) / len(recalls),
            "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "n_questions": len(recalls),
        }
        print(f"\nRecall@{k}: {metrics['recall_at_k']:.2f}")
        print(f"MRR@{k}: {metrics['mrr_at_k']:.2f}")
    else:
        print("No gold items with 'relevant_pages' found.")

    report(trackers)
    save_result("retrieval", metrics, questions, trackers)
    return metrics


if __name__ == "__main__":
    run_eval()
