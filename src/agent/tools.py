import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SQL_DB_PATH as DB_PATH
from config import SQL_DICTIONARY_PATH as DICTIONARY_PATH
from config import SQL_MAX_ROWS as MAX_ROWS


FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def load_sql_dictionary():
    return DICTIONARY_PATH.read_text(encoding="utf-8")


SQL_GUIDANCE = (
    "When writing SQL:\n"
    "- Use LIKE with wildcards for any name, keyword, or text match, never "
    "exact equality, since you do not know the exact stored spelling.\n"
    "- In datasheets_models, the model name often repeats the unit name. Do "
    "not assume a shorter generic model name exists; prefer joining on "
    "DatasheetId alone when the unit has a single model line.\n"
    "- If a query could match multiple rows, do not add extra filters to "
    "force a single result; return all matches and let the caller "
    "disambiguate.\n"
    "- Always include identifying columns (at least the datasheet name) in "
    "the SELECT.\n"
    "- Always add a LIMIT clause (20 by default) to avoid excessive rows.\n"
    "- Select only the columns the question actually needs, plus "
    "identifying columns. Avoid long text columns (description, legend) "
    "unless the question is about one specific named item in detail.\n"
)


def build_sql_instructions():
    return f"{SQL_GUIDANCE}\n{load_sql_dictionary()}"


def fix_mojibake(value):
    if not isinstance(value, str):
        return value
    try:
        fixed = value.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return value
    return fixed


def execute_sql(sql: str, max_rows: int = MAX_ROWS):
    sql = sql.strip().rstrip(";")

    if not sql.lower().startswith("select"):
        return {"error": "Only SELECT statements are allowed."}

    if FORBIDDEN_RE.search(sql):
        return {"error": "Query contains a forbidden keyword."}

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        cleaned = [{k: fix_mojibake(v) for k, v in dict(row).items()} for row in rows]
        return {"rows": cleaned, "truncated": len(rows) == max_rows}
    except sqlite3.Error as error:
        return {"error": str(error)}
    finally:
        conn.close()


def sql_query(sql: str):
    return execute_sql(sql)


if __name__ == "__main__":
    import sys

    sql = " ".join(sys.argv[1:]) or input("SQL: ")
    result = execute_sql(sql)
    print(result)
