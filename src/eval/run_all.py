import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "eval"))

import eval_retrieval
import eval_rag
import eval_sql
import eval_agent
from utils import save_manifest, start_run


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    run_dir = start_run()
    print(f"Run directory: {run_dir}")

    all_metrics = {}

    section("1/4 Retrieval")
    all_metrics["retrieval"] = eval_retrieval.run_eval()

    section("2/4 RAG generation")
    all_metrics["rag"] = eval_rag.run_eval()

    section("3/4 Text-to-SQL")
    all_metrics["sql"] = eval_sql.run_eval()

    section("4/4 Agent")
    all_metrics["agent"] = eval_agent.run_eval()

    save_manifest(all_metrics)


if __name__ == "__main__":
    main()
