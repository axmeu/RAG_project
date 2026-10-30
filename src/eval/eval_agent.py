import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from utils import LatencyTracker, latency_summary, report, save_result
from judge import judge_json

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "agent"))
sys.path.insert(0, str(SRC_DIR / "pipeline"))

import retrieve
from react_agent import run_agent
from eval_sql import expected_found
from eval_retrieval import is_relevant
from config import GEMINI_MODEL as MODEL_NAME
from config import GOLD_AGENT_PATH

load_dotenv()

VARIANTS = {
    "baseline": {},
    "no_verify": {"allow_retry": False},
    "closed_book": {"closed_book": True},
}

JUDGE_SYSTEM_INSTRUCTION = (
    "You are an independent evaluator judging whether an AI assistant's "
    "final answer correctly addresses the user's question, given the tool "
    "evidence it gathered along the way. You did not produce this answer "
    "yourself. Judge only factual correctness and completeness relative to "
    "the question and the evidence provided; ignore style. "
    "Respond with strict JSON only, no markdown fences: "
    '{"correct": true or false, "reason": "short explanation"}'
)


def judge_answer(question, evidence, answer):
    return judge_json(
        JUDGE_SYSTEM_INSTRUCTION,
        f"Question: {question}\n\n"
        f"Evidence gathered by the assistant: "
        f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Assistant's final answer: {answer}",
        {"correct": None, "reason": "judge output was not valid JSON"},
    )

def check_routing(evidence, expected_tools):
    tools_used = {step["tool"] for step in evidence}
    return set(expected_tools).issubset(tools_used), sorted(tools_used)


def check_sql(evidence, expected_rows):
    rows = []
    for step in evidence:
        if step["tool"] == "sql_query" and isinstance(step["result"], dict):
            rows.extend(step["result"].get("rows", []))
    return all(expected_found(rows, expected) for expected in expected_rows)


def check_retrieval(evidence, relevant_pages):
    chunks = []
    for step in evidence:
        if step["tool"] == "rag_search" and isinstance(step["result"], list):
            chunks.extend(step["result"])
    return any(is_relevant(chunk, relevant_pages) for chunk in chunks)


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def run_eval(variant="baseline"):
    agent_kwargs = VARIANTS[variant]
    gold = json.loads(GOLD_AGENT_PATH.read_text(encoding="utf-8"))
    client = genai.Client()
    rag_model, rag_retriever, rag_by_id = retrieve.load()
    retrieve.search(rag_model, rag_retriever, rag_by_id, gold[0]["question"])

    routing_hits = []
    sql_hits = []
    retrieval_hits = []
    answer_hits = []
    trackers = []
    questions = []

    for item in gold:
        question = item["question"]
        expected_tools = item["expected_tools"]

        tracker = LatencyTracker()
        with tracker.measure("total"):
            answer, evidence = run_agent(client,
                                         question,
                                         rag_model,
                                         rag_retriever,
                                         rag_by_id,
                                         **agent_kwargs,
                                         tracker=tracker,
                                         )
        trackers.append(tracker)

        routing_ok, tools_used = check_routing(evidence, expected_tools)
        routing_hits.append(1.0 if routing_ok else 0.0)

        print(f"\n{question}")
        print(f"   expected tools: {expected_tools}  used: {tools_used}  "
              f"routing: {'OK' if routing_ok else 'MISS'}")

        sql_ok = None
        if "expected_rows" in item:
            sql_ok = check_sql(evidence, item["expected_rows"])
            sql_hits.append(1.0 if sql_ok else 0.0)
            print(f"   SQL correctness: {'OK' if sql_ok else 'MISS'}")

        retrieval_ok = None
        if "relevant_pages" in item:
            retrieval_ok = check_retrieval(evidence, item["relevant_pages"])
            retrieval_hits.append(1.0 if retrieval_ok else 0.0)
            print(f"   retrieval: {'OK' if retrieval_ok else 'MISS'}")

        verdict = judge_answer(question, evidence, answer)
        correct = verdict.get("correct")
        if correct is not None:
            answer_hits.append(1.0 if correct else 0.0)
        print(f"   answer correctness: {correct}  ({verdict.get('reason', '')})")
        print(f"   latency: {tracker.line()}")

        questions.append(
            {
                "question": question,
                "expected_tools": expected_tools,
                "tools_used": tools_used,
                "routing_ok": routing_ok,
                "sql_ok": sql_ok,
                "retrieval_ok": retrieval_ok,
                "answer_correct": correct,
                "judge_reason": verdict.get("reason"),
                "answer": answer,
                "rag_queries": [
                    s["arguments"].get("query") for s in evidence if s["tool"] == "rag_search"
                ],
                "sql_queries": [
                    s["arguments"].get("sql") for s in evidence if s["tool"] == "sql_query"
                ],
                "latency": tracker.as_dict(),
            }
        )
    summary = latency_summary(trackers)
    metrics = {
        "variant": variant,
        "agent_kwargs": agent_kwargs,
        "latency_mean_seconds": summary["total"]["mean_seconds"],
        "latency_p95_seconds": summary["total"]["p95_seconds"],
        "routing_accuracy": mean_or_none(routing_hits),
        "routing_n": len(routing_hits),
        "sql_correctness": mean_or_none(sql_hits),
        "sql_n": len(sql_hits),
        "retrieval_recall": mean_or_none(retrieval_hits),
        "retrieval_n": len(retrieval_hits),
        "answer_correctness": mean_or_none(answer_hits),
        "answer_n": len(answer_hits),
    }

    print("\n--- Summary ---")
    for label, key, suffix in (
        ("Routing accuracy", "routing", "accuracy"),
        ("SQL correctness", "sql", "correctness"),
        ("Retrieval recall", "retrieval", "recall"),
        ("Answer correctness", "answer", "correctness"),
    ):
        value = metrics[f"{key}_{suffix}"]
        if value is not None:
            print(f"{label}: {value:.2f} ({metrics[key + '_n']} questions)")

    report(trackers)
    name = "agent" if variant == "baseline" else f"agent__{variant}"
    save_result(name, metrics, questions, trackers)
    return metrics


if __name__ == "__main__":
    run_eval()