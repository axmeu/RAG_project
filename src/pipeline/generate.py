import sys

from dotenv import load_dotenv
from google import genai

import retrieve

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"
TOP_K = 5

SYSTEM_INSTRUCTION = (
    "You answer questions about the Warhammer 40k Core Rules using only the "
    "provided context. Cite the page number(s) you used but reformulate more clearly the content."
    "Keep the specific terms of the game, do not reformulate them, except if asked clearly."
    "Give a straight answer and avoid saying that you received a context or SQL informations"
    "Be natural as if it was your own knowledge. If the context does not contain the answer,"
    "say that you do not have the answer of the question and invite to reformulate the question."
)


def build_prompt(query, contexts):
    blocks = "\n\n".join(
        f"[p.{c['page_start']}-{c['page_end']} | {c['path_str']}]\n{c['text']}"
        for c in contexts
    )
    return f"Context:\n{blocks}\n\nQuestion: {query}"


def generate(client, model, retriever, by_id, query, k=TOP_K):
    contexts = retrieve.search(model, retriever, by_id, query, k=k)
    prompt = build_prompt(query, contexts)

    interaction = client.interactions.create(
        model=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION,
        input=prompt,
    )

    return interaction.output_text, contexts


def print_answer(query, answer, contexts):
    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for c in contexts:
        print(f"   p.{c['page_start']}-{c['page_end']} {c['path_str']}")


def main():
    client = genai.Client()
    model, retriever, by_id = retrieve.load()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer, contexts = generate(client, model, retriever, by_id, query)
        print_answer(query, answer, contexts)
        return

    print("Model loaded. Type a query, or 'quit' to exit.\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        answer, contexts = generate(client, model, retriever, by_id, query)
        print()
        print_answer(query, answer, contexts)
        print()


if __name__ == "__main__":
    main()
