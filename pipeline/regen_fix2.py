#!/usr/bin/env python3
"""Regenerate ONLY closing-disclosures (40) + title-commitments (30) with
corrected prompts. Leave deeds/mortgages and all other categories untouched.
Enforce the four fixes, add a per-category document_type consistency check,
revalidate, and rebuild splits with the same seed.
"""
import json, re, time, zlib, random
from pathlib import Path
from collections import Counter, defaultdict

import generate as G

from paths import FINETUNE as FT
DOCS = FT / "docs.jsonl"
ACCEPTED = FT / "examples.jsonl"
REPORT = FT / "report.json"
SEED = G.SEED
REGEN_CATS = {"closing-disclosures", "title-commitments"}

CD_TYPE = "Closing Disclosure"
TC_TYPE = "ALTA Title Commitment"

SYSTEM = (
    "You extract fields from U.S. (Florida) real-estate closing documents. "
    "Output ONLY a single JSON object, no prose. Use EXACTLY these keys: "
    "document_type, parties (object with keys grantor, grantee, borrower, lender), "
    "property_address, legal_description, loan_amount, purchase_price, "
    "dates (object with keys executed, recorded, closing), "
    "recording_info (object with keys book, page, instrument), documentary_stamp_tax. "
    "Set a field to null when it is absent from the document. Do not invent values. "
    "PARTY ROLES: for a deed, populate parties.grantor/grantee and null parties.borrower/lender; "
    "for a mortgage, populate parties.borrower/lender and null parties.grantor/grantee. "
    "CLOSING DISCLOSURE: set document_type to exactly \"Closing Disclosure\" "
    "(never append form/TRID codes such as \"(TRID H-25)\" or \"H-25\"). "
    "TITLE COMMITMENT (ALTA Commitment for Title Insurance): set document_type to exactly "
    "\"ALTA Title Commitment\"; set purchase_price to null because the \"Proposed Amount of "
    "Insurance\" is title-insurance coverage, NOT a purchase price, so do not map it; set "
    "recording_info.book and recording_info.page to null because a plat book/page belongs to "
    "the legal description and a commitment has no recording data; set dates.executed to the "
    "Commitment Date / Effective Date. "
    "Money fields (loan_amount, purchase_price, documentary_stamp_tax) are plain JSON numbers "
    "with no currency symbol or commas. Dates are strings in YYYY-MM-DD form. "
    "recording_info values are strings. parties/property_address/legal_description/document_type are strings or null."
)

DATE_RE = re.compile(r"(?:Effective Date|Commitment Date)\s*:?\s*(\d{4}-\d{2}-\d{2})")


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


def commitment_date(text: str):
    m = DATE_RE.search(text)
    return m.group(1) if m else None


def main():
    docs = [json.loads(l) for l in open(DOCS)]
    base = [json.loads(l) for l in open(ACCEPTED)]
    assert len(docs) == len(base) == 302

    accepted = []
    discards = []
    regenerated = 0
    forced = Counter()   # what the safety net had to correct
    t0 = time.time()

    for d, ex in zip(docs, base):
        cat = d["category"]
        if cat not in REGEN_CATS:
            accepted.append((cat, ex))
            continue

        text = d["text"][:G.TEXT_CAP]
        user_content = G.USER_TMPL.format(doc=text)

        ok = None; reason = None
        for _ in range(G.MAX_RETRIES + 1):
            try:
                obj = json.loads(call(text))
            except Exception as e:
                reason = f"gen/parse error: {e}"; continue
            norm, r = G.validate_and_normalize(obj)
            if norm is None:
                reason = r; continue
            ok = norm; break

        if ok is None:
            discards.append({"filename": d["filename"], "category": cat, "reason": reason})
            continue

        # ---- enforce the four corrections (safety net over the prompt) ----
        if cat == "closing-disclosures":
            if ok["document_type"] != CD_TYPE:
                forced["cd_document_type"] += 1
            ok["document_type"] = CD_TYPE
        else:  # title-commitments
            if ok["document_type"] != TC_TYPE:
                forced["tc_document_type"] += 1
            ok["document_type"] = TC_TYPE
            if ok["purchase_price"] is not None:
                forced["tc_purchase_price_nulled"] += 1
            ok["purchase_price"] = None
            if ok["recording_info"]["book"] is not None or ok["recording_info"]["page"] is not None:
                forced["tc_recording_book_page_nulled"] += 1
            ok["recording_info"]["book"] = None
            ok["recording_info"]["page"] = None
            if ok["dates"]["executed"] is None:
                cd = commitment_date(d["text"])
                if cd:
                    ok["dates"]["executed"] = cd
                    forced["tc_executed_filled_from_text"] += 1

        example = {"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(ok, ensure_ascii=False)},
        ]}
        accepted.append((cat, example))
        regenerated += 1
        if regenerated % 10 == 0:
            print(f"[regen {regenerated}/70] {cat} elapsed={time.time()-t0:.0f}s forced={dict(forced)}", flush=True)

    with open(ACCEPTED, "w") as f:
        for _, ex in accepted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ---- per-category document_type consistency check (all categories) ----
    dt_by_cat = defaultdict(Counter)
    for cat, ex in accepted:
        dt = json.loads(ex["messages"][1]["content"])["document_type"]
        dt_by_cat[cat][dt] += 1
    KNOWN_MULTI = {"deeds"}  # legitimately >1 (Warranty Deed + Quitclaim Deed)
    consistency = {}
    failures = []
    for cat in sorted(dt_by_cat):
        types = dt_by_cat[cat]
        consistency[cat] = {"distinct": len(types), "values": dict(types)}
        if len(types) > 1 and cat not in KNOWN_MULTI:
            failures.append(cat)

    # ---- deterministic stratified 80/10/10 split (stable hash, same SEED) ----
    by_cat = defaultdict(list)
    for cat, ex in accepted:
        by_cat[cat].append(ex)
    train, valid, test = [], [], []
    for cat in sorted(by_cat):
        exs = by_cat[cat]; n = len(exs)
        idx = list(range(n))
        random.Random(SEED + zlib.crc32(cat.encode()) % 1000).shuffle(idx)
        n_test = max(1, round(n * 0.10)) if n >= 3 else 0
        n_valid = max(1, round(n * 0.10)) if n >= 3 else 0
        n_valid = min(n_valid, max(0, n - n_test - 1))
        test_idx = set(idx[:n_test]); valid_idx = set(idx[n_test:n_test + n_valid])
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
        "discarded_this_run": len(discards),
        "discards": discards,
        "safety_net_forced": dict(forced),
        "document_type_consistency": consistency,
        "consistency_failures": failures,
        "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
        "accepted_by_category": dict(Counter(c for c, _ in accepted)),
        "seed": SEED,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== REGEN2 DONE ===")
    print(f"regenerated={regenerated} discards={len(discards)} forced={dict(forced)}")
    print("document_type per category:")
    for cat in sorted(consistency):
        c = consistency[cat]
        flag = "  <-- FAIL" if (cat in failures) else (" (expected multi)" if cat in KNOWN_MULTI and c["distinct"] > 1 else "")
        print(f"  {cat}: distinct={c['distinct']} {c['values']}{flag}")
    print("consistency_failures:", failures)
    print(f"splits: train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
