import sys

from dotenv import load_dotenv
from google import genai

from tools import execute_sql, load_sql_dictionary

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"


EXTRA_GUIDANCE = (
    "Additional rules:\n"
    "- Use LIKE with wildcards for any name, keyword, or text match, never exact "
    "equality, since you do not know the exact stored spelling.\n"
    "- In datasheets_models, the model name often repeats the unit name (e.g. "
    "datasheet 'Intercessor Squad' has a model row also named 'Intercessor "
    "Squad'). Do not assume a shorter generic model name exists; prefer joining "
    "on DatasheetId alone when the unit has a single model line.\n"
    "- If a query could match multiple rows, do not add extra filters to force "
    "a single result; return all matches and let the caller disambiguate.\n"
    "- Always include identifying columns (at least the datasheet name) in the "
    "SELECT, even if not explicitly asked, so each returned row can be matched "
    "back to a specific unit.\n"
    "- Always add a LIMIT clause (20 by default, more only if the question asks "
    "for a full list) to avoid returning excessive rows.\n"
    "- Do not select long text columns (description, legend) when listing "
    "multiple rows; only select them when the question asks about one "
    "specific named item in detail.\n"
    "- Select only the columns the question actually needs, plus identifying "
    "columns. A request for 'which stratagems' needs only stratagem names "
    "(and the datasheet name for context), not cost, phase, or type unless "
    "asked.\n"
)


def build_system_instruction():
    return (
        "You translate natural language questions about Warhammer 40k units "
        "into a single SQLite SELECT query, using the schema described below. "
        "Return only the SQL query, no explanation, no markdown fences.\n\n"
        f"{EXTRA_GUIDANCE}\n{load_sql_dictionary()}"
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
