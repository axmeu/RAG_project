import sys
from pathlib import Path

import gradio as gr
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "pipeline"))

import retrieve
from react_agent import run_agent

CLIENT = genai.Client()
RAG_MODEL, RAG_RETRIEVER, RAG_BY_ID = retrieve.load()


def respond(message, history):
    answer, evidence = run_agent(
        CLIENT, message, RAG_MODEL, RAG_RETRIEVER, RAG_BY_ID
    )
    tools_used = ", ".join(sorted({step["tool"] for step in evidence})) or "none"
    return f"{answer}\n\n_Tools used: {tools_used}_"


demo = gr.ChatInterface(
    fn=respond,
    title="Warhammer 40k Rules Assistant",
    description=(
        "Ask about Core Rules mechanics or unit stats/stratagems. "
        "The agent decides whether to search the rulebook, query the "
        "unit database, or both."
    ),
    examples=[
        "How does the charge phase work?",
        "What is the toughness of a Space Marine Intercessor?",
        "What stratagems can I use during the shooting phase?",
    ],
)

if __name__ == "__main__":
    demo.launch()
