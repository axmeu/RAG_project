import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "agent"))

from tools import build_sql_instructions, execute_sql
from config import GEMINI_MODEL as MODEL_NAME
from config import GOLD_SQL_PATH


load_dotenv()

SQL_GEN_SYSTEM_INSTRUCTION = (
    "You translate natural language questions about Warhammer 40k units "
    "into a single SQLite SELECT query, using the schema described below. "
    "Return only the SQL query, no explanation, no markdown fences.\n\n"
    f"{build_sql_instructions()}"
)


def generate_sql(client, question):
    interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=SQL_GEN_SYSTEM_INSTRUCTION,
        input=question,
    )
    return interaction.output_text.strip()


def row_matches(row, expected):
    row_values = {str(v).strip().lower() for v in row.values()}
    return all(
        str(value).strip().lower() in row_values for value in expected.values()
    )


def expected_found(rows, expected):
    return any(row_matches(row, expected) for row in rows)


def run_eval():
    gold = json.loads(GOLD_SQL_PATH.read_text(encoding="utf-8"))
    client = genai.Client()

    hits = 0
    for item in gold:
        question = item["question"]
        expected_rows = item["expected_rows"]

        sql = generate_sql(client, question)
        result = execute_sql(sql)
        rows = result.get("rows", [])

        all_found = all(expected_found(rows, expected) for expected in expected_rows)
        hits += int(all_found)

        print(f"{question}")
        print(f"   SQL: {sql}")
        print(f"   {'OK' if all_found else 'MISS'} ({len(rows)} rows returned)")
        if not all_found:
            print(f"   expected: {expected_rows}")

    print(f"\nAccuracy: {hits}/{len(gold)}")


if __name__ == "__main__":
    run_eval()
