import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

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

JUDGE_SYSTEM_INSTRUCTION = (
    "You are an independent evaluator judging whether an AI assistant's "
    "final answer correctly addresses the user's question, given the tool "
    "evidence it gathered along the way. You did not produce this answer "
    "yourself. Judge only factual correctness and completeness relative to "
    "the question and the evidence provided; ignore style. "
    "Respond with strict JSON only, no markdown fences: "
    '{"correct": true or false, "reason": "short explanation"}'
)


def judge_answer(client, question, evidence, answer):
    interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=JUDGE_SYSTEM_INSTRUCTION,
        input=(
            f"Question: {question}\n\n"
            f"Evidence gathered by the assistant: "
            f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
            f"Assistant's final answer: {answer}"
        ),
    )
    try:
        return json.loads(interaction.output_text.strip())
    except json.JSONDecodeError:
        return {"correct": None, "reason": "judge output was not valid JSON"}


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


def run_eval():
    gold = json.loads(GOLD_AGENT_PATH.read_text(encoding="utf-8"))
    client = genai.Client()
    rag_model, rag_retriever, rag_by_id = retrieve.load()

    routing_hits = []
    sql_hits = []
    retrieval_hits = []
    answer_hits = []

    for item in gold:
        question = item["question"]
        expected_tools = item["expected_tools"]

        answer, evidence = run_agent(
            client, question, rag_model, rag_retriever, rag_by_id
        )

        routing_ok, tools_used = check_routing(evidence, expected_tools)
        routing_hits.append(1.0 if routing_ok else 0.0)

        print(f"\n{question}")
        print(f"   expected tools: {expected_tools}  used: {tools_used}  "
              f"routing: {'OK' if routing_ok else 'MISS'}")

        if "expected_rows" in item:
            sql_ok = check_sql(evidence, item["expected_rows"])
            sql_hits.append(1.0 if sql_ok else 0.0)
            print(f"   SQL correctness: {'OK' if sql_ok else 'MISS'}")

        if "relevant_pages" in item:
            retrieval_ok = check_retrieval(evidence, item["relevant_pages"])
            retrieval_hits.append(1.0 if retrieval_ok else 0.0)
            print(f"   retrieval: {'OK' if retrieval_ok else 'MISS'}")

        verdict = judge_answer(client, question, evidence, answer)
        correct = verdict.get("correct")
        if correct is not None:
            answer_hits.append(1.0 if correct else 0.0)
        print(f"   answer correctness: {correct}  ({verdict.get('reason', '')})")

    print("\n--- Summary ---")
    if routing_hits:
        print(f"Routing accuracy: {sum(routing_hits) / len(routing_hits):.2f} "
              f"({len(routing_hits)} questions)")
    if sql_hits:
        print(f"SQL correctness: {sum(sql_hits) / len(sql_hits):.2f} "
              f"({len(sql_hits)} questions)")
    if retrieval_hits:
        print(f"Retrieval recall: {sum(retrieval_hits) / len(retrieval_hits):.2f} "
              f"({len(retrieval_hits)} questions)")
    if answer_hits:
        print(f"Answer correctness: {sum(answer_hits) / len(answer_hits):.2f} "
              f"({len(answer_hits)} questions)")


if __name__ == "__main__":
    run_eval()