#!/usr/bin/env python3
"""Generate structured-extraction training examples from corpus text using
qwen3:32b via Ollama, validate against a fixed schema, discard nonconforming,
and split 80/10/10 into train/valid/test (mlx-lm chat format).
"""
import json, sys, time, urllib.request, random
from pathlib import Path
from collections import Counter, defaultdict

from paths import FINETUNE as FT
DOCS = FT / "docs.jsonl"
ACCEPTED = FT / "examples.jsonl"
REPORT = FT / "report.json"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3:32b"
TEXT_CAP = 6000          # cap embedded document text (bounds latency; affects only large real-forms)
MAX_RETRIES = 1          # regenerate once before discarding
SEED = 20260829

USER_TMPL = "Extract the key fields from this document as JSON:\n\n{doc}"
SYSTEM = (
    "You extract fields from U.S. (Florida) real-estate closing documents. "
    "Output ONLY a single JSON object, no prose. Use EXACTLY these keys: "
    "document_type, parties (object with keys grantor, grantee, borrower, lender), "
    "property_address, legal_description, loan_amount, purchase_price, "
    "dates (object with keys executed, recorded, closing), "
    "recording_info (object with keys book, page, instrument), documentary_stamp_tax. "
    "Set a field to null when it is absent from the document. Do not invent values. "
    "Money fields (loan_amount, purchase_price, documentary_stamp_tax) are plain JSON numbers "
    "with no currency symbol or commas. Dates are strings in YYYY-MM-DD form. "
    "recording_info values are strings. parties/property_address/legal_description/document_type are strings or null."
)

TOP_KEYS = {"document_type","parties","property_address","legal_description",
            "loan_amount","purchase_price","dates","recording_info","documentary_stamp_tax"}
PARTY_KEYS = {"grantor","grantee","borrower","lender"}
DATE_KEYS = {"executed","recorded","closing"}
REC_KEYS = {"book","page","instrument"}
STR_OR_NULL = ("document_type","property_address","legal_description")
MONEY = ("loan_amount","purchase_price","documentary_stamp_tax")


def call_ollama(doc_text: str) -> str:
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(doc=doc_text)},
        ],
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp["message"]["content"]


def _num_or_null(v):
    if v is None: return None, True
    if isinstance(v, bool): return None, False
    if isinstance(v, (int, float)): return v, True
    if isinstance(v, str):
        s = v.strip().replace("$", "").replace(",", "")
        if s == "" or s.lower() == "null": return None, True
        try:
            f = float(s); return (int(f) if f.is_integer() else f), True
        except ValueError:
            return None, False
    return None, False


def _str_or_null(v):
    if v is None: return None, True
    if isinstance(v, (int, float)): return str(v), True
    if isinstance(v, str):
        s = v.strip()
        return (None if s == "" or s.lower() == "null" else s), True
    return None, False


def validate_and_normalize(obj):
    """Return (normalized_obj, None) if conforming else (None, reason)."""
    if not isinstance(obj, dict):
        return None, "not an object"
    if set(obj.keys()) != TOP_KEYS:
        return None, f"top-level keys mismatch: {sorted(obj.keys())}"
    out = {}
    for k in STR_OR_NULL:
        v, ok = _str_or_null(obj[k])
        if not ok: return None, f"{k} not str/null"
        out[k] = v
    for k in MONEY:
        v, ok = _num_or_null(obj[k])
        if not ok: return None, f"{k} not number/null"
        out[k] = v
    p = obj["parties"]
    if not isinstance(p, dict) or set(p.keys()) != PARTY_KEYS:
        return None, "parties keys mismatch"
    out["parties"] = {}
    for k in PARTY_KEYS:
        v, ok = _str_or_null(p[k])
        if not ok: return None, f"parties.{k} not str/null"
        out["parties"][k] = v
    d = obj["dates"]
    if not isinstance(d, dict) or set(d.keys()) != DATE_KEYS:
        return None, "dates keys mismatch"
    out["dates"] = {}
    for k in DATE_KEYS:
        v, ok = _str_or_null(d[k])
        if not ok: return None, f"dates.{k} not str/null"
        out["dates"][k] = v
    r = obj["recording_info"]
    if not isinstance(r, dict) or set(r.keys()) != REC_KEYS:
        return None, "recording_info keys mismatch"
    out["recording_info"] = {}
    for k in REC_KEYS:
        v, ok = _str_or_null(r[k])
        if not ok: return None, f"recording_info.{k} not str/null"
        out["recording_info"][k] = v
    # canonical key order
    ordered = {
        "document_type": out["document_type"],
        "parties": {k: out["parties"][k] for k in ("grantor","grantee","borrower","lender")},
        "property_address": out["property_address"],
        "legal_description": out["legal_description"],
        "loan_amount": out["loan_amount"],
        "purchase_price": out["purchase_price"],
        "dates": {k: out["dates"][k] for k in ("executed","recorded","closing")},
        "recording_info": {k: out["recording_info"][k] for k in ("book","page","instrument")},
        "documentary_stamp_tax": out["documentary_stamp_tax"],
    }
    return ordered, None


def main():
    docs = [json.loads(l) for l in open(DOCS)]
    random.seed(SEED)

    accepted = []            # list of (category, example_dict)
    discards = []            # list of (filename, reason)
    truncated = 0
    t0 = time.time()

    for i, d in enumerate(docs, 1):
        text = d["text"]
        if len(text) > TEXT_CAP:
            text = text[:TEXT_CAP]
            truncated += 1
        user_content = USER_TMPL.format(doc=text)

        ok_obj = None
        last_reason = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = call_ollama(text)
            except Exception as e:
                last_reason = f"ollama error: {e}"
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                last_reason = f"json parse error: {e}"
                continue
            norm, reason = validate_and_normalize(obj)
            if norm is not None:
                ok_obj = norm
                break
            last_reason = reason

        if ok_obj is None:
            discards.append({"filename": d["filename"], "category": d["category"], "reason": last_reason})
        else:
            assistant_content = json.dumps(ok_obj, ensure_ascii=False)
            example = {"messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]}
            accepted.append((d["category"], example))

        if i % 10 == 0 or i == len(docs):
            el = time.time() - t0
            rate = el / i
            eta = rate * (len(docs) - i)
            print(f"[{i}/{len(docs)}] accepted={len(accepted)} discarded={len(discards)} "
                  f"elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    # write all accepted examples (order preserved)
    with open(ACCEPTED, "w") as f:
        for _, ex in accepted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # stratified 80/10/10 split by category
    by_cat = defaultdict(list)
    for cat, ex in accepted:
        by_cat[cat].append(ex)
    train, valid, test = [], [], []
    for cat, exs in by_cat.items():
        idx = list(range(len(exs)))
        random.Random(SEED + hash(cat) % 1000).shuffle(idx)
        n = len(exs)
        n_test = max(1, round(n * 0.10)) if n >= 3 else 0
        n_valid = max(1, round(n * 0.10)) if n >= 3 else 0
        # ensure train keeps the majority
        n_valid = min(n_valid, max(0, n - n_test - 1))
        test_idx = set(idx[:n_test])
        valid_idx = set(idx[n_test:n_test + n_valid])
        for j, ex in enumerate(exs):
            if j in test_idx: test.append(ex)
            elif j in valid_idx: valid.append(ex)
            else: train.append(ex)

    random.Random(SEED).shuffle(train)
    random.Random(SEED + 1).shuffle(valid)
    random.Random(SEED + 2).shuffle(test)

    def dump(name, rows):
        with open(FT / name, "w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dump("train.jsonl", train)
    dump("valid.jsonl", valid)
    dump("test.jsonl", test)

    total = len(docs)
    discard_rate = len(discards) / total if total else 0.0
    report = {
        "model": MODEL,
        "total_documents": total,
        "accepted": len(accepted),
        "discarded": len(discards),
        "discard_rate": round(discard_rate, 4),
        "truncated_docs_over_cap": truncated,
        "text_cap_chars": TEXT_CAP,
        "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
        "accepted_by_category": dict(Counter(cat for cat, _ in accepted)),
        "discards": discards,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== DONE ===")
    print(f"total={total} accepted={len(accepted)} discarded={len(discards)} "
          f"discard_rate={discard_rate*100:.2f}%")
    print(f"splits: train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"truncated docs (> {TEXT_CAP} chars): {truncated}")
    print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
