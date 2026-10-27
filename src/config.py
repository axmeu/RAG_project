from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEMINI_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "answerdotai/answerai-colbert-small-v1"
OCR_MODEL = "lightonai/LightOnOCR-3-0.8B"  # requires flashattention on recent gpu or slow
INDEX_NAME = "core_rules"

RAG_TOP_K = 5
SQL_MAX_ROWS = 200
AGENT_MAX_ITERATIONS = 5

PDF_PATH = ROOT / "data" / "Core Rules.pdf"
OCR_DIR = ROOT / "data" / "ocr"
STRUCTURED_DIR = ROOT / "data" / "structured"
SECTIONS_PATH = STRUCTURED_DIR / "sections.json"
CHUNKS_PATH = STRUCTURED_DIR / "chunks.json"
INDEX_DIR = ROOT / "data" / "index"
SQL_DB_PATH = ROOT / "data" / "wahadb.sqlite"
SQL_DICTIONARY_PATH = ROOT / "src" / "agent" / "SQL_AGENT_DATA_DICTIONARY.md"
