import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SECTIONS_PATH = ROOT / "data" / "structured" / "sections.json"
OUT_PATH = ROOT / "data" / "structured" / "chunks.json"

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 400
MAX_TOKENS = 700
MIN_TOKENS = 80

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN
MIN_CHARS = MIN_TOKENS * CHARS_PER_TOKEN

PARA_RE = re.compile(r"\n\s*\n")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def n_tokens(text):
    return len(text) // CHARS_PER_TOKEN


def path_str(section):
    return " > ".join(section["path"])


def pages_for_span(spans, start, end):
    page_start = page_end = None
    for span in spans:
        if span["end"] <= start or span["start"] >= end:
            continue
        if page_start is None:
            page_start = span["page"]
        page_end = span["page"]
    if page_start is None:
        page_start = page_end = spans[0]["page"] if spans else None
    return page_start, page_end


def split_paragraphs(text):
    pieces = []
    offset = 0
    for m in PARA_RE.finditer(text):
        pieces.append((offset, m.start()))
        offset = m.end()
    pieces.append((offset, len(text)))
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def split_sentences(text, start):
    pieces = []
    offset = 0
    for m in SENT_RE.finditer(text):
        pieces.append((start + offset, start + m.start()))
        offset = m.end()
    pieces.append((start + offset, start + len(text)))
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def pack_spans(spans, max_chars):
    groups, current_start, current_end = [], None, None
    for start, end in spans:
        if current_start is None:
            current_start, current_end = start, end
            continue
        if end - current_start <= max_chars:
            current_end = end
        else:
            groups.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None:
        groups.append((current_start, current_end))
    return groups


def section_spans(text):
    paras = split_paragraphs(text)
    spans = []
    for start, end in paras:
        if end - start <= MAX_CHARS:
            spans.append((start, end))
            continue
        for s, e in split_sentences(text[start:end], start):
            spans.append((s, e))
    return spans


def make_chunk(section, start, end, chunk_id):
    text = section["text"][start:end].strip()
    page_start, page_end = pages_for_span(section["page_spans"], start, end)
    return {
        "chunk_id": chunk_id,
        "section_id": section["id"],
        "chapter_id": section.get("chapter_id"),
        "title": section["title"],
        "path_str": path_str(section),
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
    }


def chunk_section(section, chunk_id_start):
    text = section["text"]
    chunk_id = chunk_id_start
    chunks = []

    if n_tokens(text) <= MAX_TOKENS:
        chunks.append(make_chunk(section, 0, len(text), f"c{chunk_id:04d}"))
        return chunks, chunk_id + 1

    spans = section_spans(text)
    groups = pack_spans(spans, MAX_CHARS)
    for start, end in groups:
        chunks.append(make_chunk(section, start, end, f"c{chunk_id:04d}"))
        chunk_id += 1
    return chunks, chunk_id


def attach_chapter_ids(sections):
    chapter_for_id = {}
    for s in sections:
        if s["kind"] in ("part", "chapter"):
            chapter_for_id[s["id"]] = s["id"] if s["kind"] == "chapter" else None

    cache = {}

    def resolve(section_id, by_id):
        if section_id in cache:
            return cache[section_id]
        sec = by_id[section_id]
        if sec["kind"] == "chapter":
            result = sec["id"]
        elif sec["parent"] == "doc":
            result = None
        else:
            result = resolve(sec["parent"], by_id)
        cache[section_id] = result
        return result

    by_id = {s["id"]: s for s in sections}
    for s in sections:
        s["chapter_id"] = resolve(s["id"], by_id)


def merge_short_chunks(chunks):
    merged = []
    for c in chunks:
        if merged and n_tokens(c["text"]) < MIN_TOKENS and c["chapter_id"] == merged[-1]["chapter_id"]:
            prev = merged[-1]
            prev["text"] = prev["text"] + "\n\n" + c["text"]
            prev["page_end"] = c["page_end"]
            prev.setdefault("merged_section_ids", [prev["section_id"]])
            prev["merged_section_ids"].append(c["section_id"])
        else:
            merged.append(c)
    return merged


def main():
    sections = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    attach_chapter_ids(sections)

    leaf_sections = []
    for i, s in enumerate(sections):
        is_leaf = i + 1 == len(sections) or sections[i + 1]["level"] <= s["level"]
        if is_leaf:
            leaf_sections.append(s)

    chunks, chunk_id = [], 1
    for s in leaf_sections:
        if not s["text"].strip():
            continue
        new_chunks, chunk_id = chunk_section(s, chunk_id)
        chunks.extend(new_chunks)

    chunks = merge_short_chunks(chunks)

    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    sizes = [n_tokens(c["text"]) for c in chunks]
    print(f"Chunks: {len(chunks)}")
    print(f"Tokens: min={min(sizes)} max={max(sizes)} avg={sum(sizes) // len(sizes)}")
    print(f"Chunks over {MAX_TOKENS} tokens:", sum(1 for s in sizes if s > MAX_TOKENS))
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
