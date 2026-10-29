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


FONTS_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""

CSS = """
:root {
    --bg: #1a1611;
    --panel: #221d16;
    --ink: #e9e1cd;
    --ink-dim: #a99d82;
    --gold: #c9a227;
    --red: #7c2320;
    --line: #4a4030;
    html, body, gradio-app {
    background: var(--bg) !important;
    }
}

.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
}

#hero {
    text-align: center;
    padding: 2.25rem 1rem 1.25rem 1rem;
}

#hero h1 {
    font-family: 'Cinzel', serif !important;
    font-weight: 800 !important;
    font-size: 2.1rem !important;
    letter-spacing: 0.04em;
    color: var(--gold) !important;
    margin: 0 0 0.6rem 0 !important;
    text-transform: uppercase;
}

#hero p {
    color: var(--ink-dim) !important;
    font-size: 0.95rem !important;
    line-height: 1.5;
    max-width: 520px;
    margin: 0 auto !important;
}

#hero::after {
    content: "";
    display: block;
    width: 120px;
    height: 2px;
    background: var(--gold);
    margin: 1.1rem auto 0 auto;
    opacity: 0.6;
}

#chatbot {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
}

button.primary {
    background: var(--red) !important;
    border-color: var(--red) !important;
}
"""

with gr.Blocks(css=CSS, head=FONTS_HEAD, theme=gr.themes.Base()) as demo:
    with gr.Column(elem_id="hero"):
        gr.Markdown(
            "# Warhammer 40k Rules Assistant\n\n"
            "Ask about Core Rules mechanics or unit stats and stratagems. "
            "The agent decides whether to search the rulebook, query the "
            "unit database, or both."
        )

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(elem_id="chatbot", height=480),
        examples=[
            "How does the charge phase work?",
            "What is the toughness of a Space Marine Intercessor?",
            "What stratagems can I use during the shooting phase?",
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)
