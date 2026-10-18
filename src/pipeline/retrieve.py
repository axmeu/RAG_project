import json
import sys
from pathlib import Path
from pylate import indexes, models
from pylate import retrieve


ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_PATH = ROOT / "data" / "structured" / "chunks.json"
INDEX_DIR = ROOT / "data" / "index"

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
INDEX_NAME = "core_rules"
TOP_K = 5


def load():
    model = models.ColBERT(model_name_or_path=MODEL_NAME)
    index = indexes.PLAID(index_folder=str(INDEX_DIR), index_name=INDEX_NAME)
    retriever = retrieve.ColBERT(index=index)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}
    return model, retriever, by_id


def search(model, retriever, by_id, query, k=TOP_K):
    query_embedding = model.encode(
        [query],
        batch_size=1,
        is_query=True,
        show_progress_bar=False,
    )
    results = retriever.retrieve(
        queries_embeddings=query_embedding,
        k=k,
    )[0]
    return [{**by_id[r["id"]], "score": r["score"]} for r in results]


def main():
    query = " ".join(sys.argv[1:]) or input("Query: ")
    model, retriever, by_id = load()
    results = search(model, retriever, by_id, query)

    print(f"Query: {query}\n")
    for r in results:
        print(f"score={r['score']:.2f} p.{r['page_start']}-{r['page_end']} {r['path_str']}")
        print(r["text"][:300].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
