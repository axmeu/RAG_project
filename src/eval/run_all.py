import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "eval"))

import eval_retrieval
import eval_rag
import eval_sql
import eval_agent


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


if __name__ == "__main__":
    section("RETRIEVAL")
    eval_retrieval.run_eval()

    section("RAG (faithfulness / relevancy)")
    eval_rag.run_eval()

    section("SQL")
    eval_sql.run_eval()

    section("AGENT (routing / tool correctness / answer)")
    eval_agent.run_eval()