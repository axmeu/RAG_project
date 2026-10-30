import json
from functools import lru_cache
import anthropic
from dotenv import load_dotenv
from config import JUDGE_MODEL

load_dotenv()


@lru_cache(maxsize=1)
def get_client():
    return anthropic.Anthropic()


def judge_json(system_instruction, user_input, fallback):
    message = get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=4096,
        system=system_instruction,
        messages=[{"role": "user", "content": user_input}],
    )
    text = "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return fallback
