import json
from functools import lru_cache
from dotenv import load_dotenv
from config import JUDGE_MODEL, JUDGE_PROVIDER


load_dotenv()


@lru_cache(maxsize=1)
def get_client():
    if JUDGE_PROVIDER == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    if JUDGE_PROVIDER == "gemini":
        from google import genai

        return genai.Client()
    raise ValueError(f"Unknown JUDGE_PROVIDER: {JUDGE_PROVIDER}")


def complete(system_instruction, user_input):
    client = get_client()
    if JUDGE_PROVIDER == "anthropic":
        message = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4096,
            system=system_instruction,
            messages=[{"role": "user", "content": user_input}],
        )
        return "".join(
            block.text for block in message.content if block.type == "text"
        )
    interaction = client.interactions.create(
        model=JUDGE_MODEL,
        system_instruction=system_instruction,
        input=user_input,
    )
    return interaction.output_text


def judge_json(system_instruction, user_input, fallback):
    text = complete(system_instruction, user_input).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return fallback