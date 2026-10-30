import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "pipeline"))

import retrieve
import eval_agent
from eval_retrieval import is_relevant
from utils import LatencyTracker, latency_summary, report, save_result, start_run, write_json
from config import GOLD_AGENT_PATH, RAG_TOP_K


load_dotenv()


def run_raw_retrieval():
    gold = json.loads(GOLD_AGENT_PATH.read_text(encoding="utf-8"))
    model, retriever, by_id = retrieve.load()
    retrieve.search(model, retriever, by_id, gold[0]["question"], k=RAG_TOP_K)

    hits = []
    trackers = []
    questions = []
    for item in gold:
        if "relevant_pages" not in item:
            continue

        tracker = LatencyTracker()
        with tracker.measure("total"), tracker.measure("retrieval"):
            results = retrieve.search(
                model, retriever, by_id, item["question"], k=RAG_TOP_K
            )
        trackers.append(tracker)

        hit = any(is_relevant(chunk, item["relevant_pages"]) for chunk in results)
        hits.append(1.0 if hit else 0.0)
        questions.append(
            {
                "question": item["question"],
                "relevant_pages": item["relevant_pages"],
                "retrieved_pages": [[c["page_start"], c["page_end"]] for c in results],
                "retrieval_ok": hit,
                "latency": tracker.as_dict(),
            }
        )
        print(f"{item['question']}\n   raw query retrieval: {'OK' if hit else 'MISS'}")

    summary = latency_summary(trackers)
    metrics = {
        "variant": "raw_retrieval",
        "retrieval_recall": sum(hits) / len(hits),
        "retrieval_n": len(hits),
        "latency_mean_seconds": summary["total"]["mean_seconds"],
        "latency_p95_seconds": summary["total"]["p95_seconds"],
    }
    print(f"\nRaw-query retrieval recall: {metrics['retrieval_recall']:.2f} ({len(hits)} questions)")
    report(trackers)
    save_result("agent__raw_retrieval", metrics, questions, trackers)
    return metrics


def format_value(value):
    if value is None:
        return "-"
    return f"{value:.2f}"


def print_table(results):
    columns = (
        ("routing_accuracy", "routing"),
        ("sql_correctness", "sql"),
        ("retrieval_recall", "retrieval"),
        ("answer_correctness", "answer"),
        ("latency_mean_seconds", "mean s"),
        ("latency_p95_seconds", "p95 s"),
    )
    header = f"{'variant':<16}" + "".join(f"{label:>11}" for _, label in columns)
    print("\n" + header)
    print("-" * len(header))
    for variant, metrics in results.items():
        row = f"{variant:<16}" + "".join(
            f"{format_value(metrics.get(key)):>11}" for key, _ in columns
        )
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[*eval_agent.VARIANTS, "raw_retrieval"],
        choices=[*eval_agent.VARIANTS, "raw_retrieval"],
    )
    args = parser.parse_args()

    start_run()
    results = {}
    for variant in args.variants:
        print(f"\n{'=' * 60}\nAblation: {variant}\n{'=' * 60}")
        if variant == "raw_retrieval":
            results[variant] = run_raw_retrieval()
        else:
            results[variant] = eval_agent.run_eval(variant)

    print_table(results)
    path = start_run() / "ablations.json"
    write_json(path, results)
    print(f"\nAblation summary saved to {path}")


if __name__ == "__main__":
    main()
