import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from config import GEMINI_MODEL as MODEL_NAME
from tools import build_sql_instructions, execute_sql


AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR.parent))

load_dotenv()


def build_system_instruction():
    return (
        "You translate natural language questions about Warhammer 40k units "
        "into a single SQLite SELECT query, using the schema described below. "
        "Return only the SQL query, no explanation, no markdown fences.\n\n"
        f"{build_sql_instructions()}"
    )


ANSWER_SYSTEM_INSTRUCTION = (
    "You answer questions about Warhammer 40k using only the provided SQL "
    "query result. If the result is empty or contains an error, say so "
    "explicitly instead of guessing."
)


def ask(client, question):
    sql_interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=build_system_instruction(),
        input=question,
    )
    sql = sql_interaction.output_text.strip()
    result = execute_sql(sql)

    print(f"Question: {question}")
    print(f"SQL: {sql}")
    print(f"Result: {result}\n")

    answer_interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=ANSWER_SYSTEM_INSTRUCTION,
        input=f"Question: {question}\n\nSQL result: {result}",
    )
    print(f"Answer:\n{answer_interaction.output_text}")


def main():
    question = " ".join(sys.argv[1:]) or input("Question: ")
    client = genai.Client()
    ask(client, question)


if __name__ == "__main__":
    main()
