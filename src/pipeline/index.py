import json
import sys
from pathlib import Path
from pylate import indexes, models, retrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNKS_PATH, EMBEDDING_MODEL, INDEX_DIR, INDEX_NAME

MODEL_NAME = EMBEDDING_MODEL


def embedding_text(chunk):
    return chunk["path_str"] + "\n\n" + chunk["text"]


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    model = models.ColBERT(model_name_or_path=MODEL_NAME)
    index = indexes.PLAID(
        index_folder=str(INDEX_DIR),
        index_name=INDEX_NAME,
        override=True,
    )

    documents_ids = [c["chunk_id"] for c in chunks]
    documents = [embedding_text(c) for c in chunks]

    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
    )

    print(f"Indexed {len(chunks)} chunks into {INDEX_DIR} (index={INDEX_NAME})")


if __name__ == "__main__":
    main()
