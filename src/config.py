from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent.parent

# Models
GEMINI_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "answerdotai/answerai-colbert-small-v1"
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "anthropic")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")
OCR_MODEL = "lightonai/LightOnOCR-3-0.8B"  # requires flashattention on recent gpu or slow

# PDF doc
INDEX_NAME = "core_rules"

# Configs
RAG_TOP_K = 5
SQL_MAX_ROWS = 200
AGENT_MAX_ITERATIONS = 5

# Paths
PDF_PATH = ROOT / "data" / "Core Rules.pdf"
OCR_DIR = ROOT / "data" / "ocr"
STRUCTURED_DIR = ROOT / "data" / "structured"
SECTIONS_PATH = STRUCTURED_DIR / "sections.json"
CHUNKS_PATH = STRUCTURED_DIR / "chunks.json"
INDEX_DIR = ROOT / "data" / "index"
SQL_DB_PATH = ROOT / "data" / "raw" / "wahadb.sqlite"
SQL_DICTIONARY_PATH = ROOT / "src" / "agent" / "SQL_AGENT_DATA_DICTIONARY.md"

# Eval Path
EVAL_DIR = ROOT / "data" / "eval"
GOLD_RAG_PATH = EVAL_DIR / "gold_rag.json"
GOLD_SQL_PATH = EVAL_DIR / "gold_sql.json"
GOLD_AGENT_PATH = EVAL_DIR / "gold_agent.json"
