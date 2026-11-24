#!/usr/bin/env python3
"""Regenerate ONLY deeds (90) + mortgages (80) with corrected party-mapping
prompt, revalidate, keep all other categories' examples unchanged, and rebuild
the 80/10/10 splits deterministically with the same seed.
"""
import json, time, zlib, random
from pathlib import Path
from collections import Counter, defaultdict

import generate as G  # reuse call_ollama, validate_and_normalize, USER_TMPL, TEXT_CAP, SEED

from paths import FINETUNE as FT
DOCS = FT / "docs.jsonl"
ACCEPTED = FT / "examples.jsonl"
REPORT = FT / "report.json"
SEED = G.SEED
REGEN_CATS = {"deeds", "mortgages"}

# Corrected system prompt: explicit, content-based party-role rule.
SYSTEM = (
    "You extract fields from U.S. (Florida) real-estate closing documents. "
    "Output ONLY a single JSON object, no prose. Use EXACTLY these keys: "
    "document_type, parties (object with keys grantor, grantee, borrower, lender), "
    "property_address, legal_description, loan_amount, purchase_price, "
    "dates (object with keys executed, recorded, closing), "
    "recording_info (object with keys book, page, instrument), documentary_stamp_tax. "
    "Set a field to null when it is absent from the document. Do not invent values. "
    "PARTY ROLES: If the document is a DEED (e.g. warranty deed, quitclaim deed) that "
    "conveys title between a grantor and a grantee, populate parties.grantor and "
    "parties.grantee and set parties.borrower and parties.lender to null. "
    "If the document is a MORTGAGE or security instrument between a borrower/mortgagor "
    "and a lender/mortgagee, populate parties.borrower and parties.lender and set "
    "parties.grantor and parties.grantee to null. "
    "Money fields (loan_amount, purchase_price, documentary_stamp_tax) are plain JSON numbers "
    "with no currency symbol or commas. Dates are strings in YYYY-MM-DD form. "
    "recording_info values are strings. parties/property_address/legal_description/document_type are strings or null."
)


def call(doc_text: str) -> str:
    import urllib.request
    body = json.dumps({
        "model": G.MODEL, "stream": False, "think": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": G.USER_TMPL.format(doc=doc_text)},
        ],
    }).encode()
    req = urllib.request.Request(G.OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]


def main():
    docs = [json.loads(l) for l in open(DOCS)]
    base = [json.loads(l) for l in open(ACCEPTED)]  # 302, doc order, 1:1 with docs
    assert len(docs) == len(base) == 302

    accepted = []          # (category, example)
    discards = []
    forced = 0             # times the categorical invariant had to be enforced
    regenerated = 0
    t0 = time.time()

    for i, (d, ex) in enumerate(zip(docs, base), 1):
        cat = d["category"]
        if cat not in REGEN_CATS:
            accepted.append((cat, ex))
            continue

        text = d["text"][:G.TEXT_CAP]
        user_content = G.USER_TMPL.format(doc=text)
        is_deed = (cat == "deeds")

        ok = None
        reason = None
        for _ in range(G.MAX_RETRIES + 1):
            try:
                raw = call(text)
                obj = json.loads(raw)
            except Exception as e:
                reason = f"gen/parse error: {e}"; continue
            norm, r = G.validate_and_normalize(obj)
            if norm is None:
                reason = r; continue
            ok = norm; break

        if ok is None:
            discards.append({"filename": d["filename"], "category": cat, "reason": reason})
            continue

        # Enforce the categorical party invariant (safety net over the prompt).
        p = ok["parties"]
        if is_deed:
            if p["borrower"] is not None or p["lender"] is not None:
                forced += 1
            p["borrower"] = None; p["lender"] = None
        else:  # mortgage
            if p["grantor"] is not None or p["grantee"] is not None:
                forced += 1
            p["grantor"] = None; p["grantee"] = None

        example = {"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(ok, ensure_ascii=False)},
        ]}
        accepted.append((cat, example))
        regenerated += 1
        if regenerated % 10 == 0:
            el = time.time() - t0
            print(f"[regen {regenerated}/170] {cat} elapsed={el:.0f}s forced_so_far={forced}", flush=True)

    # write examples.jsonl (all, doc order)
    with open(ACCEPTED, "w") as f:
        for _, ex in accepted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # deterministic stratified 80/10/10 split (stable hash; same SEED)
    by_cat = defaultdict(list)
    for cat, ex in accepted:
        by_cat[cat].append(ex)
    train, valid, test = [], [], []
    for cat in sorted(by_cat):
        exs = by_cat[cat]
        n = len(exs)
        idx = list(range(n))
        stable = zlib.crc32(cat.encode()) % 1000
        random.Random(SEED + stable).shuffle(idx)
        n_test = max(1, round(n * 0.10)) if n >= 3 else 0
        n_valid = max(1, round(n * 0.10)) if n >= 3 else 0
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
    dump("train.jsonl", train); dump("valid.jsonl", valid); dump("test.jsonl", test)

    report = {
        "model": G.MODEL,
        "regenerated_categories": sorted(REGEN_CATS),
        "regenerated_docs": regenerated,
        "party_invariant_forced": forced,
        "total_documents": len(docs),
        "accepted": len(accepted),
        "discarded_this_run": len(discards),
        "discards": discards,
        "text_cap_chars": G.TEXT_CAP,
        "max_synthetic_doc_chars": max(d["chars"] for d in docs if d["category"] != "real-forms"),
        "synthetic_docs_truncated": 0,
        "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
        "accepted_by_category": dict(Counter(c for c, _ in accepted)),
        "seed": SEED,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== REGEN DONE ===")
    print(f"regenerated={regenerated} (deeds+mortgages) discards_this_run={len(discards)} "
          f"party_invariant_forced={forced}")
    print(f"splits: train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
