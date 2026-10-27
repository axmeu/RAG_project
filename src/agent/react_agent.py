import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai


SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "pipeline"))

import retrieve
from tools import build_sql_instructions, execute_sql
from config import AGENT_MAX_ITERATIONS as MAX_ITERATIONS
from config import GEMINI_MODEL as MODEL_NAME
from config import RAG_TOP_K


load_dotenv()

RAG_SEARCH_TOOL = {
    "type": "function",
    "name": "rag_search",
    "description": (
        "Search the Warhammer 40k Core Rules text for rules, mechanics, "
        "phases, and game procedures (e.g. how charging works, what "
        "happens on a morale check, how cover works). Does not contain "
        "unit stats, points costs, abilities, or stratagems."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query over the rules text.",
            }
        },
        "required": ["query"],
    },
}

SQL_QUERY_TOOL = {
    "type": "function",
    "name": "sql_query",
    "description": (
        "Execute a SQLite SELECT query against the unit database "
        "(datasheets, model stats, abilities, keywords, stratagems, "
        "enhancements, factions, detachments). Does not contain general "
        "rules text. Write the SQL yourself using the schema given in "
        "your instructions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single SQLite SELECT statement.",
            }
        },
        "required": ["sql"],
    },
}

TOOLS = [RAG_SEARCH_TOOL, SQL_QUERY_TOOL]

RAG_GUIDANCE = (
    "When calling rag_search:\n"
    "- Never pass the user's question verbatim. First reason about what is "
    "actually being asked in terms of Warhammer 40k game mechanics: a "
    "situation described in plain, colloquial language can involve one or "
    "several distinct rules systems (phases, actions, keywords, unit "
    "stats) that are not named explicitly in the question. Work out every "
    "distinct mechanic that could be relevant before searching, based on "
    "your own knowledge of how the game works, not on a fixed list.\n"
    "- For each mechanic you identify, issue a separate rag_search query, "
    "phrased using the official Warhammer 40k rules terminology for that "
    "mechanic, not the user's own wording.\n"
    "- If the results do not clearly answer the question, reconsider "
    "whether a relevant mechanic was missed, reformulate, and search again "
    "before answering from memory or giving up.\n"
)

SYSTEM_INSTRUCTION = (
    "You are an assistant answering questions about Warhammer 40k. You have "
    "two tools: rag_search for rules and mechanics, sql_query for unit "
    "stats and reference data. Decide which tool(s) a question needs based "
    "on their descriptions. A question can require both tools in sequence. "
    "Cite page numbers when you use rag_search results. Do not answer from "
    "memory; only use what the tools return. If the tools do not contain "
    "the answer, say so explicitly.\n\n"
    f"{RAG_GUIDANCE}\n"
    f"{build_sql_instructions()}"
)

VERIFY_SYSTEM_INSTRUCTION = (
    "You are a critical reviewer, independent from the assistant that "
    "produced the answer below. Given the original question, the evidence "
    "gathered, and the candidate answer, decide if the answer fully and "
    "accurately addresses the question using only that evidence. "
    "Respond with strict JSON only, no markdown fences: "
    '{"satisfied": true or false, "missing": "what is missing, or empty '
    'string if satisfied"}'
)


def dispatch_tool(name, arguments, rag_model, rag_retriever, rag_by_id):
    if name == "rag_search":
        contexts = retrieve.search(
            rag_model, rag_retriever, rag_by_id, arguments["query"], k=RAG_TOP_K
        )
        return [
            {
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "path_str": c["path_str"],
                "text": c["text"],
            }
            for c in contexts
        ]

    if name == "sql_query":
        return execute_sql(arguments["sql"])

    return {"error": f"Unknown tool: {name}"}


def run_tool_loop(client, question, rag_model, rag_retriever, rag_by_id, evidence):
    interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        input=question,
        tools=TOOLS,
    )

    for _ in range(MAX_ITERATIONS):
        function_calls = [s for s in interaction.steps if s.type == "function_call"]
        if not function_calls:
            break

        for call in function_calls:
            result = dispatch_tool(
                call.name, call.arguments, rag_model, rag_retriever, rag_by_id
            )
            evidence.append({"tool": call.name, "arguments": call.arguments, "result": result})

            interaction = client.interactions.create(
                model=MODEL_NAME,
                previous_interaction_id=interaction.id,
                tools=TOOLS,
                input=[{
                    "type": "function_result",
                    "name": call.name,
                    "call_id": call.id,
                    "result": json.dumps(result, ensure_ascii=False),
                }],
            )

    return interaction


def verify_answer(client, question, evidence, answer):
    interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=VERIFY_SYSTEM_INSTRUCTION,
        input=(
            f"Question: {question}\n\n"
            f"Evidence gathered: {json.dumps(evidence, ensure_ascii=False)}\n\n"
            f"Candidate answer: {answer}"
        ),
    )
    try:
        return json.loads(interaction.output_text.strip())
    except json.JSONDecodeError:
        return {"satisfied": True, "missing": ""}


def run_agent(client, question):
    evidence = []
    rag_model, rag_retriever, rag_by_id = retrieve.load()

    current_question = question
    final_answer = None

    for round_index in range(MAX_ITERATIONS):
        interaction = run_tool_loop(
            client, current_question, rag_model, rag_retriever, rag_by_id, evidence
        )
        final_answer = interaction.output_text

        verdict = verify_answer(client, question, evidence, final_answer)
        if verdict.get("satisfied", True):
            return final_answer, evidence

        missing = verdict.get("missing", "")
        current_question = (
            f"{question}\n\n"
            f"Your previous answer was incomplete. Missing: {missing}. "
            "Use the tools again to gather what is missing, then answer again."
        )

    return final_answer, evidence


def main():
    question = " ".join(sys.argv[1:]) or input("Question: ")
    client = genai.Client()
    answer, evidence = run_agent(client, question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")
    print(f"Tool calls made: {len(evidence)}")
    for step in evidence:
        print(f"   {step['tool']}({step['arguments']})")


if __name__ == "__main__":
    main()
