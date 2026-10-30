import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from utils import LatencyTracker, latency_summary, save_result
from judge import judge_json


SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "pipeline"))

import retrieve
from generate import generate
from config import GEMINI_MODEL as MODEL_NAME
from config import GOLD_RAG_PATH


load_dotenv()

JUDGE_SYSTEM_INSTRUCTION = (
    "You are an evaluator judging a RAG system's answer. Given a question, "
    "the context it was given, and the answer it produced, score two things:\n"
    "- faithfulness: the fraction of factual claims in the answer that are "
    "directly supported by the context (0.0 to 1.0). List any claim that is "
    "not supported.\n"
    "- answer_relevancy: how directly the answer addresses the question "
    "(0.0 to 1.0), independent of whether it is factually supported.\n"
    "Respond with strict JSON only, no markdown fences: "
    '{"faithfulness": 0.0, "unsupported_claims": [], '
    '"answer_relevancy": 0.0, "relevancy_note": ""}'
)


def judge(question, contexts, answer):
    context_text = "\n\n".join(
        f"[p.{c['page_start']}-{c['page_end']}]\n{c['text']}" for c in contexts
    )
    return judge_json(
        JUDGE_SYSTEM_INSTRUCTION,
        f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer: {answer}",
        {
            "faithfulness": None,
            "unsupported_claims": [],
            "answer_relevancy": None,
            "relevancy_note": "judge output was not valid JSON",
        },
    )


def run_eval():
    gold = json.loads(GOLD_RAG_PATH.read_text(encoding="utf-8"))
    client = genai.Client()
    model, retriever, by_id = retrieve.load()

    results = []
    trackers = []
    for item in gold:
        question = item["question"]

        tracker = LatencyTracker()
        with tracker.measure("total"):
            answer, contexts = generate(client, model, retriever, by_id, question)
        trackers.append(tracker)

        verdict = judge(question, contexts, answer)
        results.append(
            {
                "question": question,
                "answer": answer,
                **verdict,
                "latency": tracker.as_dict(),
            }
        )

        print(f"{question}")
        print(
            f"   faithfulness: {verdict.get('faithfulness')}  "
            f"answer_relevancy: {verdict.get('answer_relevancy')}"
        )
        if verdict.get("unsupported_claims"):
            print(f"   unsupported: {verdict['unsupported_claims']}")

    scored = [r for r in results if r.get("faithfulness") is not None]
    metrics = {"n_questions": len(results), "n_scored": len(scored)}
    if scored:
        metrics["avg_faithfulness"] = sum(r["faithfulness"] for r in scored) / len(scored)
        metrics["avg_answer_relevancy"] = sum(r["answer_relevancy"] for r in scored) / len(scored)
        print(f"\nAverage faithfulness: {metrics['avg_faithfulness']:.2f}")
        print(f"Average answer_relevancy: {metrics['avg_answer_relevancy']:.2f}")

    summary = latency_summary(trackers)
    print(f"Mean generation latency: {summary['total']['mean_seconds']:.2f}s")

    save_result("rag", metrics, results, trackers)
    return metrics


if __name__ == "__main__":
    run_eval()
