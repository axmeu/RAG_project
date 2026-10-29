import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

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


def judge(client, question, contexts, answer):
    context_text = "\n\n".join(
        f"[p.{c['page_start']}-{c['page_end']}]\n{c['text']}" for c in contexts
    )
    interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=JUDGE_SYSTEM_INSTRUCTION,
        input=(
            f"Question: {question}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer: {answer}"
        ),
    )
    try:
        return json.loads(interaction.output_text.strip())
    except json.JSONDecodeError:
        return {
            "faithfulness": None,
            "unsupported_claims": [],
            "answer_relevancy": None,
            "relevancy_note": "judge output was not valid JSON",
        }


def run_eval():
    gold = json.loads(GOLD_RAG_PATH.read_text(encoding="utf-8"))
    client = genai.Client()
    model, retriever, by_id = retrieve.load()

    results = []
    for item in gold:
        question = item["question"]
        answer, contexts = generate(client, model, retriever, by_id, question)
        verdict = judge(client, question, contexts, answer)
        results.append({"question": question, "answer": answer, **verdict})

        print(f"{question}")
        print(
            f"   faithfulness: {verdict.get('faithfulness')}  "
            f"answer_relevancy: {verdict.get('answer_relevancy')}"
        )
        if verdict.get("unsupported_claims"):
            print(f"   unsupported: {verdict['unsupported_claims']}")

    scored = [r for r in results if r.get("faithfulness") is not None]
    if scored:
        avg_faithfulness = sum(r["faithfulness"] for r in scored) / len(scored)
        avg_relevancy = sum(r["answer_relevancy"] for r in scored) / len(scored)
        print(f"\nAverage faithfulness: {avg_faithfulness:.2f}")
        print(f"Average answer_relevancy: {avg_relevancy:.2f}")

    return results


if __name__ == "__main__":
    run_eval()
