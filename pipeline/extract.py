#!/usr/bin/env python3
"""Extract plain text from every document in the corpus -> docs.jsonl
Each line: {"path": str, "category": str, "filename": str, "text": str}
"""
import json, sys
from pathlib import Path
import fitz  # pymupdf
import docx

from paths import CORPUS, FINETUNE

OUT = FINETUNE / "docs.jsonl"

def pdf_text(p: Path) -> str:
    parts = []
    with fitz.open(p) as d:
        for page in d:
            parts.append(page.get_text("text"))
    return "\n".join(parts)

def docx_text(p: Path) -> str:
    d = docx.Document(str(p))
    parts = [para.text for para in d.paragraphs]
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [c.text for c in row.cells]
            parts.append("\t".join(cells))
    return "\n".join(parts)

def main():
    rows = []
    for p in sorted(CORPUS.rglob("*")):
        if p.is_dir() or p.name.startswith("."):
            continue
        if ".vectorstore" in p.parts:
            continue
        ext = p.suffix.lower()
        try:
            if ext == ".pdf":
                text = pdf_text(p)
            elif ext == ".docx":
                text = docx_text(p)
            else:
                continue
        except Exception as e:
            print(f"ERR {p}: {e}", file=sys.stderr)
            continue
        text = text.strip()
        rows.append({
            "path": str(p),
            "category": p.parent.name,
            "filename": p.name,
            "text": text,
            "chars": len(text),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # summary
    from collections import Counter
    c = Counter(r["category"] for r in rows)
    empty = sum(1 for r in rows if r["chars"] < 30)
    print(f"extracted {len(rows)} docs; empty/short(<30 chars)={empty}")
    for k in sorted(c):
        cats = [r["chars"] for r in rows if r["category"] == k]
        print(f"  {k}: n={c[k]} avg_chars={sum(cats)//len(cats)}")

if __name__ == "__main__":
    main()
