import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = ROOT / "data" / "ocr"
OUT_DIR = ROOT / "data" / "structured"

BRAND = "CORE RULES"
EXTRA_CHAPTERS = []
MIN_TEXT_CHARS = 15
CHAPTER_MATCH = 0.85
MAX_LEVEL = 6
SHORT_SECTION = 30
TOC_MIN_ENTRIES = 3
FOOTER_LOOKBACK = 15

COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*$")
SOURCE_RE = re.compile(r"<!--\s*Source:\s*(.*?)\s*-->")
PAGE_RE = re.compile(r"<!--\s*Page:\s*(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)(?:\s*\d+,\d+,\d+,\d+)?")
OCR_NOTE_RE = re.compile(
    r"^note:.*\b(sidebar|layout context|not included|visual element|tabular structure)\b", re.I
)
TOC_RE = re.compile(
    r"^(?P<title>.+?)\s*\(\s*(?:pgs?|pp?|pages?)\.?\s*(?P<start>\d+)(?:\s*[-–—]\s*(?P<end>\d+))?\s*\)\s*$",
    re.I,
)

CAPS = r"[A-Z0-9][A-Z0-9 '\-]{1,60}[A-Z0-9]"
FOOTER_RE = re.compile(rf"^(?P<a>{CAPS})\s*\|\s*(?P<b>{CAPS})$")


def key(s):
    return " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())


def clean_title(s):
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", s)).strip()


def same_title(a, b):
    return SequenceMatcher(None, key(a), key(b)).ratio() >= CHAPTER_MATCH


def parse_footer(lines):
    tail = [i for i, l in enumerate(lines) if l.strip()][-FOOTER_LOOKBACK:]
    for i in reversed(tail):
        s = re.sub(r"[*_`]", "", lines[i]).strip().strip("|# ").strip()
        m = FOOTER_RE.match(s)
        if not m:
            continue
        a, b = m.group("a").strip(), m.group("b").strip()
        if same_title(a, BRAND):
            return i, b, s
        if same_title(b, BRAND):
            return i, a, s
    return None


def read_page(path):
    raw = path.read_text(encoding="utf-8")
    source = SOURCE_RE.search(raw)
    number = PAGE_RE.search(raw)
    page = int(number.group(1)) if number else int(re.search(r"\d+", path.stem).group())
    raw_lines = [l.rstrip() for l in raw.splitlines() if not COMMENT_RE.match(l)]
    n_notes = sum(1 for l in raw_lines if OCR_NOTE_RE.match(l.strip()))
    lines = [l for l in raw_lines if not OCR_NOTE_RE.match(l.strip())]

    chapter = footer_raw = None
    n_trailing = 0
    footer = parse_footer(lines)
    if footer:
        idx, chapter, footer_raw = footer
        n_trailing = len(lines) - idx
        lines = lines[:idx]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    body = "\n".join(lines)
    images = IMAGE_RE.findall(body)
    alnum = sum(c.isalnum() for c in IMAGE_RE.sub("", body))
    if alnum >= MIN_TEXT_CHARS:
        kind = "text"
    else:
        kind = "image_only" if images else "empty"

    headings = []
    for idx, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if not m:
            continue
        text = clean_title(m.group(2))
        if text:
            headings.append(
                {"line": idx, "text": text, "local_level": len(m.group(1)), "dropped": False, "toc": False}
            )

    return {
        "page": page,
        "file": path.name,
        "source": source.group(1) if source else None,
        "printed_page": page,
        "footer_raw": footer_raw,
        "part": None,
        "part_source": None,
        "chapter": chapter,
        "chapter_source": "footer" if chapter else None,
        "chapter_uncertain": False,
        "switch_line": None,
        "kind": kind,
        "n_chars": len(body),
        "n_images": len(images),
        "n_table_rows": sum(1 for l in lines if l.lstrip().startswith("|")),
        "n_ocr_notes": n_notes,
        "n_trailing_lines_dropped": n_trailing,
        "ends_with_heading": bool(headings) and headings[-1]["line"] == len(lines) - 1,
        "headings": headings,
        "lines": lines,
        "text": body,
    }


def canonicalize_chapters(pages):
    counts = Counter(p["chapter"] for p in pages if p["chapter"])
    canon, mapping = [], {}
    for name, _ in counts.most_common():
        target = next((c for c in canon if same_title(name, c)), None)
        if target is None:
            canon.append(name)
            target = name
        mapping[name] = target
    for p in pages:
        if p["chapter"]:
            p["chapter"] = mapping[p["chapter"]]


def parse_toc(pages, offset):
    parts = []
    for p in pages:
        entries = [(h, TOC_RE.match(h["text"])) for h in p["headings"]]
        entries = [(h, m) for h, m in entries if m]
        if len(entries) < TOC_MIN_ENTRIES:
            continue
        for h, m in entries:
            start = int(m.group("start"))
            end = int(m.group("end") or start)
            h["toc"] = True
            parts.append(
                {
                    "title": clean_title(m.group("title")),
                    "page_start": start - offset,
                    "page_end": end - offset,
                    "toc_page": p["page"],
                }
            )
    return parts


def assign_parts(pages, parts):
    for p in pages:
        for part in parts:
            if part["page_start"] <= p["page"] <= part["page_end"]:
                p["part"], p["part_source"] = part["title"], "toc"
                break


def chapter_part_map(pages):
    votes = defaultdict(Counter)
    names = []
    for p in pages:
        if p["chapter_source"] != "footer":
            continue
        if p["chapter"] not in names:
            names.append(p["chapter"])
        if p["part"]:
            votes[p["chapter"]][p["part"]] += 1
    return {n: (votes[n].most_common(1)[0][0] if votes[n] else None) for n in names}


def match_chapter(text, known):
    best, score = None, 0.0
    for name in known:
        r = SequenceMatcher(None, key(text), key(name)).ratio()
        if r > score:
            best, score = name, r
    return best if score >= CHAPTER_MATCH else None


def known_chapters(pages):
    return sorted({p["chapter"] for p in pages if p["chapter"]} | set(EXTRA_CHAPTERS))


def valid_for(p, chapter, cmap):
    if chapter is None or p["part"] is None:
        return True
    if chapter in EXTRA_CHAPTERS and chapter not in cmap:
        cmap[chapter] = p["part"]
    return cmap.get(chapter) == p["part"]


def resolve_chapters(pages, cmap):
    known = known_chapters(pages)
    for i, p in enumerate(pages):
        if p["chapter"]:
            continue
        prev_ch = next((q["chapter"] for q in reversed(pages[:i]) if q["chapter"]), None)
        next_ch = next((q["chapter"] for q in pages[i + 1:] if q["chapter"]), None)
        if prev_ch is not None and prev_ch == next_ch and valid_for(p, prev_ch, cmap):
            p["chapter"], p["chapter_source"] = prev_ch, "interval"
            continue
        plain = [h for h in p["headings"] if not h["toc"]]
        top = min((h["local_level"] for h in plain), default=None)
        target = line = None
        for h in plain:
            c = match_chapter(h["text"], known)
            confirmed = next_ch is None or c == next_ch or c in EXTRA_CHAPTERS
            if c and c != prev_ch and h["local_level"] == top and confirmed and valid_for(p, c, cmap):
                target, line = c, h["line"]
                break
        if target:
            p["chapter"], p["switch_line"], p["chapter_source"] = target, line, "heading"
        elif valid_for(p, prev_ch, cmap):
            p["chapter"], p["chapter_source"] = prev_ch, "carried"
            p["chapter_uncertain"] = (
                prev_ch is not None
                and next_ch is not None
                and prev_ch != next_ch
                and prev_ch not in EXTRA_CHAPTERS
            )
        else:
            p["chapter"], p["chapter_source"] = None, "part"


def finalize_parts(pages, cmap):
    for p in pages:
        if p["part"] is None and p["chapter"] and cmap.get(p["chapter"]):
            p["part"], p["part_source"] = cmap[p["chapter"]], "chapter"


def chain_for(part, chapter):
    if part and chapter and not same_title(part, chapter):
        return [(part, "part"), (chapter, "chapter")]
    if part:
        return [(part, "part")]
    if chapter:
        return [(chapter, "chapter")]
    return []


def build_sections(pages, parts):
    toc = {part["title"]: [part["page_start"], part["page_end"]] for part in parts}
    sections, stack = [], []

    def touch(page):
        for s in stack:
            s["page_end"] = page

    def open_section(level, title, page, kind):
        level = min(level, MAX_LEVEL)
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        level = min(level, stack[-1]["level"] + 1) if stack else 1
        sec = {
            "id": f"s{len(sections) + 1:03d}",
            "kind": kind,
            "level": level,
            "title": title,
            "parent": stack[-1]["id"] if stack else "doc",
            "path": [s["title"] for s in stack] + [title],
            "page_start": page,
            "page_end": page,
            "lines": [],
        }
        if kind == "part" and title in toc:
            sec["toc_pages"] = toc[title]
        sections.append(sec)
        stack.append(sec)
        touch(page)
        return sec

    def set_chain(chain, page):
        current = sum(1 for s in stack if s["kind"] != "section")
        depth = 0
        while (
            depth < len(chain)
            and depth < current
            and stack[depth]["title"] == chain[depth][0]
        ):
            depth += 1
        if not (depth == len(chain) == current):
            del stack[depth:]
            for i in range(depth, len(chain)):
                open_section(i + 1, chain[i][0], page, chain[i][1])

    for p in pages:
        switch = p["switch_line"]
        chain = chain_for(p["part"], p["chapter"])
        if switch is None:
            set_chain(chain, p["page"])
        else:
            keep = stack and (not p["part"] or stack[0]["title"] == p["part"])
            if keep:
                start = [(s["title"], s["kind"]) for s in stack if s["kind"] != "section"]
            else:
                start = chain_for(p["part"], None)
            set_chain(start, p["page"])
        chain_keys = {key(t) for t in (p["part"], p["chapter"]) if t}

        heads = {h["line"]: h for h in p["headings"]}
        out_lines = []
        for idx, line in enumerate(p["lines"]):
            h = heads.get(idx)
            if h is None:
                stack[-1]["lines"].append((p["page"], line))
                if line.strip():
                    touch(p["page"])
                out_lines.append(line)
                continue
            if h["toc"]:
                plain = re.sub(r"^#+\s*", "", line.strip())
                stack[-1]["lines"].append((p["page"], plain))
                touch(p["page"])
                out_lines.append(plain)
                h.update(level=None, requested=None)
                continue
            if switch is not None and idx == switch:
                set_chain(chain, p["page"])
                h.update(level=1, requested=1, dropped=True)
                continue
            if key(h["text"]) in chain_keys:
                h.update(level=1, requested=1, dropped=True)
                continue
            base = sum(1 for s in stack if s["kind"] != "section")
            requested = h["local_level"] + base
            sec = open_section(requested, h["text"], p["page"], "section")
            h.update(level=sec["level"], requested=requested)
            out_lines.append("#" * sec["level"] + " " + h["text"])
        p["text"] = "\n".join(out_lines).strip()

    for sec in sections:
        lines = sec.pop("lines")
        while lines and not lines[0][1].strip():
            lines.pop(0)
        while lines and not lines[-1][1].strip():
            lines.pop()
        spans, offset = [], 0
        for page, line in lines:
            end = offset + len(line)
            if spans and spans[-1]["page"] == page:
                spans[-1]["end"] = end
            else:
                spans.append({"page": page, "start": offset, "end": end})
            offset = end + 1
        sec["text"] = "\n".join(l for _, l in lines)
        sec["page_spans"] = spans
        sec["n_chars"] = len(sec["text"])
    return sections


def write_outline(sections, path):
    with path.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(
                "  " * (s["level"] - 1)
                + f"- {s['title']} (p.{s['page_start']}-{s['page_end']}, {s['n_chars']} chars)\n"
            )


def build_graph(sections, pages):
    n_pages, n_footer = Counter(), Counter()
    for p in pages:
        names = {p["part"], p["chapter"]} - {None}
        for name in names:
            n_pages[name] += 1
            n_footer[name] += p["chapter_source"] == "footer"

    parent = {s["id"]: s["parent"] for s in sections}
    descendants = Counter()
    for s in sections:
        if s["kind"] == "section":
            a = s["parent"]
            while a != "doc":
                descendants[a] += 1
                a = parent[a]

    nodes = [{"id": "doc", "type": "document", "title": pages[0]["source"] or "document"}]
    for s in sections:
        node = {
            "id": s["id"],
            "type": s["kind"],
            "title": s["title"],
            "level": s["level"],
            "parent": s["parent"],
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "n_chars": s["n_chars"],
        }
        if s["kind"] != "section":
            node["n_pages"] = n_pages[s["title"]]
            node["footer_pages"] = n_footer[s["title"]]
            node["n_sections"] = descendants[s["id"]]
        if "toc_pages" in s:
            node["toc_pages"] = s["toc_pages"]
        nodes.append(node)

    edges, last = [], {}
    for s in sections:
        edges.append({"source": s["parent"], "target": s["id"], "type": "contains"})
        if s["parent"] in last:
            edges.append({"source": last[s["parent"]], "target": s["id"], "type": "next"})
        last[s["parent"]] = s["id"]
    return {"nodes": nodes, "edges": edges}


def report(pages, sections, graph, parts):
    nodes = graph["nodes"]
    structural = [n for n in nodes if n["type"] in ("part", "chapter")]
    print(f"Pages: {len(pages)} | Sections: {len(sections)} | Parts/chapters: {len(structural)}")
    print("Page kinds:", dict(Counter(p["kind"] for p in pages)))

    numbers = [p["page"] for p in pages]
    missing = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
    print("Missing page numbers:", missing or "none")
    print("Sources:", {p["source"] for p in pages})

    notes = sum(p["n_ocr_notes"] for p in pages)
    note_pages = [p["page"] for p in pages if p["n_ocr_notes"]]
    print(f"OCR self-referential notes stripped: {notes} | pages: {note_pages or 'none'}")
    trailing = sum(p["n_trailing_lines_dropped"] for p in pages)
    trailing_pages = [p["page"] for p in pages if p["n_trailing_lines_dropped"]]
    print(f"Lines dropped after a detected footer: {trailing} on {len(trailing_pages)} pages")

    print(f"Table of contents entries: {len(parts)}")
    covered = set()
    for part in parts:
        covered.update(range(part["page_start"], part["page_end"] + 1))
        found = next((n for n in nodes if n["type"] == "part" and n["title"] == part["title"]), None)
        status = f"observed p.{found['page_start']}-{found['page_end']}" if found else "NOT FOUND IN PAGES"
        print(f"   {part['title']:<38} toc p.{part['page_start']}-{part['page_end']} | {status}")
    outside = [n for n in numbers if n not in covered]
    print("Pages outside any table of contents range:", outside or "none")

    print("Chapter sources:", dict(Counter(p["chapter_source"] or "none" for p in pages)))
    print("Pages without a footer:", [p["page"] for p in pages if p["chapter_source"] != "footer"] or "none")
    switches = [f"p.{p['page']} -> {p['chapter']}" for p in pages if p["chapter_source"] == "heading"]
    print("Chapter changes detected from a heading (no footer):", switches or "none")
    print(
        "Pages with an uncertain chapter (between two different footer chapters):",
        [p["page"] for p in pages if p["chapter_uncertain"]] or "none",
    )

    spread = defaultdict(set)
    for p in pages:
        if p["chapter_source"] == "footer" and p["part"]:
            spread[p["chapter"]].add(p["part"])
    multi = {c: sorted(v) for c, v in spread.items() if len(v) > 1}
    print("Footer chapters spanning several table of contents parts:", multi or "none")

    print("Structure graph (parts and chapters):")
    for n in structural:
        toc = f" toc={n['toc_pages'][0]}-{n['toc_pages'][1]}" if "toc_pages" in n else ""
        print(
            f"   {'  ' * (n['level'] - 1)}{n['id']} [{n['type']}] {n['title']} "
            f"p.{n['page_start']}-{n['page_end']} pages={n['n_pages']} footer={n['footer_pages']} "
            f"sections={n['n_sections']}{toc}"
        )
    titles = Counter(n["title"] for n in structural)
    print("Nodes appearing several times:", [t for t, c in titles.items() if c > 1] or "none")

    known = known_chapters(pages)
    foreign = []
    for p in pages:
        for h in p["headings"]:
            c = None if (h["dropped"] or h["toc"]) else match_chapter(h["text"], known)
            if c and c != p["chapter"]:
                foreign.append(f"p.{p['page']} '{h['text']}' looks like chapter {c} (page chapter: {p['chapter']})")
    print(f"Headings named like another chapter: {len(foreign)}")
    for line in foreign[:20]:
        print("  ", line)

    openers = []
    for p in pages:
        plain = [h for h in p["headings"] if not h["toc"]]
        if p["chapter_source"] not in ("carried", "part") or not plain:
            continue
        top = min(h["local_level"] for h in plain)
        first = next(h for h in plain if h["local_level"] == top)
        if first["text"].isupper() and not match_chapter(first["text"], known):
            openers.append(f"p.{p['page']} {first['text']}")
    print(f"Top headings of footer-less pages that are not known chapters (see EXTRA_CHAPTERS): {len(openers)}")
    for line in openers[:30]:
        print("  ", line)

    kept = [h for p in pages for h in p["headings"] if not h["dropped"] and not h["toc"]]
    clamped = [h for h in kept if h["level"] != h["requested"]]
    dropped = sum(h["dropped"] for p in pages for h in p["headings"])
    print(f"Headings kept: {len(kept)} | dropped as part/chapter titles: {dropped}")
    print(f"Headings whose level was clamped to keep a valid tree: {len(clamped)}")
    print("Sections per level:", dict(sorted(Counter(s["level"] for s in sections).items())))

    short = []
    for i, s in enumerate(sections):
        is_leaf = i + 1 == len(sections) or sections[i + 1]["level"] <= s["level"]
        if is_leaf and s["n_chars"] < SHORT_SECTION:
            short.append(f"{s['id']} p.{s['page_start']} {s['title']}")
    print(f"Very short leaf sections: {len(short)}")
    for line in short[:20]:
        print("  ", line)

    paths = Counter(tuple(s["path"]) for s in sections)
    dupes = [" > ".join(k) for k, c in paths.items() if c > 1]
    print(f"Duplicate section paths: {len(dupes)}")
    for d in dupes[:20]:
        print("  ", d)

    print(
        "Pages ending with a heading (section continues on next page):",
        [p["page"] for p in pages if p["ends_with_heading"]] or "none",
    )


def main():
    files = sorted(INPUT_DIR.glob("page_*.md"))
    if not files:
        print(f"No page_*.md files found in {INPUT_DIR}")
        return

    pages = sorted((read_page(f) for f in files), key=lambda p: p["page"])
    canonicalize_chapters(pages)
    parts = parse_toc(pages, 0)
    assign_parts(pages, parts)
    cmap = chapter_part_map(pages)
    resolve_chapters(pages, cmap)
    finalize_parts(pages, cmap)
    sections = build_sections(pages, parts)
    graph = build_graph(sections, pages)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sections.json").write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
    write_outline(sections, OUT_DIR / "outline.md")

    report(pages, sections, graph, parts)
    print(f"Written to {OUT_DIR}: sections.json, outline.md")


if __name__ == "__main__":
    main()
