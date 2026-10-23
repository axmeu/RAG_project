import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "data" / "raw" / "wahadb.sqlite"
DICTIONARY_PATH = Path(__file__).resolve().parent / "SQL_AGENT_DATA_DICTIONARY.md"

MAX_ROWS = 200

FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def load_sql_dictionary():
    return DICTIONARY_PATH.read_text(encoding="utf-8")


def fix_mojibake(value):
    # fixing the encoding artefacts
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
