import json
from pathlib import Path

from pylate import indexes, models, retrieve

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_PATH = ROOT / "data" / "structured" / "chunks.json"
INDEX_DIR = ROOT / "data" / "index"

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
INDEX_NAME = "core_rules"

SANITY_QUERIES = [
    "What happens when a unit falls back?",
    "How does the charge phase work?",
    "What is engagement range?"
]


def embedding_text(chunk):
    # concatenates the title path with the chunk
    return chunk["path_str"] + "\n\n" + chunk["text"]


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    model = models.ColBERT(model_name_or_path=MODEL_NAME)

    # plaid with index override as we wont add more data in this project
    index = indexes.PLAID(
        index_folder=str(INDEX_DIR),
        index_name=INDEX_NAME,
        override=True
    )

    documents_ids = [c["chunk_id"] for c in chunks]
    documents = [embedding_text(c) for c in chunks]

    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,
        show_progress_bar=True
    )

    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings
    )

    print(f"Indexed {len(chunks)} chunks into {INDEX_DIR} (index={INDEX_NAME})")

    retriever = retrieve.ColBERT(index=index)
    by_id = {c["chunk_id"]: c for c in chunks}

    for query in SANITY_QUERIES:
        query_embedding = model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )
        results = retriever.retrieve(queries_embeddings=query_embedding, k=3)[0]

        print(f"\nQuery: {query}")
        for r in results:
            c = by_id[r["id"]]
            print(f"   score={r['score']:.2f} p.{c['page_start']} {c['path_str']}")


if __name__ == "__main__":
    main()
