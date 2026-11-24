#!/usr/bin/env python3
"""Evaluate a model (optionally with a LoRA adapter) on the closing-document
structured-extraction test set via mlx-lm.

Reports:
  * Valid-JSON rate       - % of outputs that parse as JSON
  * Schema-conformance     - % with the exact expected key structure and types
  * Field-level accuracy   - % of leaf fields matching the reference, overall
                             and per field (null-vs-null counts as a match)

Per-example results are written to a JSON file for failure inspection.

Usage:
  python eval.py --model mlx-community/Qwen3-8B-4bit
  python eval.py --model mlx-community/Qwen3-8B-4bit --adapter-path adapters/
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Written by pipeline/generate.py (see pipeline/paths.py). Override with --test.
DEFAULT_TEST = Path(__file__).resolve().parent / "work" / "finetune" / "test.jsonl"

# Built-in system prompt describing the exact target schema. Selected with
# --system-prompt SCHEMA. Fully specifies key names, nesting, types, the
# null-for-absent rule, and plain-number money formatting.
SCHEMA_SYSTEM_PROMPT = (
    "You extract fields from U.S. (Florida) real-estate closing documents. "
    "Output ONLY a single JSON object and nothing else. The object must have "
    "EXACTLY these keys, with this exact nesting:\n"
    "{\n"
    '  "document_type": string | null,\n'
    '  "parties": {\n'
    '    "grantor": string | null,\n'
    '    "grantee": string | null,\n'
    '    "borrower": string | null,\n'
    '    "lender": string | null\n'
    "  },\n"
    '  "property_address": string | null,\n'
    '  "legal_description": string | null,\n'
    '  "loan_amount": number | null,\n'
    '  "purchase_price": number | null,\n'
    '  "dates": {\n'
    '    "executed": string | null,\n'
    '    "recorded": string | null,\n'
    '    "closing": string | null\n'
    "  },\n"
    '  "recording_info": {\n'
    '    "book": string | null,\n'
    '    "page": string | null,\n'
    '    "instrument": string | null\n'
    "  },\n"
    '  "documentary_stamp_tax": number | null\n'
    "}\n"
    "Rules: Use null (JSON null, not the string \"null\") for any field that is "
    "absent from the document; do not invent values. Money fields (loan_amount, "
    "purchase_price, documentary_stamp_tax) must be plain JSON numbers with no "
    "currency symbol, commas, or quotes. Dates are strings in YYYY-MM-DD form. "
    "All recording_info values are strings or null. Do not add, remove, or rename "
    "any keys, and do not nest fields differently than shown."
)

# ---- schema definition ------------------------------------------------------
TOP_KEYS = {"document_type", "parties", "property_address", "legal_description",
            "loan_amount", "purchase_price", "dates", "recording_info",
            "documentary_stamp_tax"}
PARTY_KEYS = {"grantor", "grantee", "borrower", "lender"}
DATE_KEYS = {"executed", "recorded", "closing"}
REC_KEYS = {"book", "page", "instrument"}
STR_OR_NULL = ("document_type", "property_address", "legal_description")
MONEY = ("loan_amount", "purchase_price", "documentary_stamp_tax")

# Ordered leaf paths used for field-level accuracy.
LEAF_FIELDS = [
    "document_type",
    "parties.grantor", "parties.grantee", "parties.borrower", "parties.lender",
    "property_address", "legal_description",
    "loan_amount", "purchase_price",
    "dates.executed", "dates.recorded", "dates.closing",
    "recording_info.book", "recording_info.page", "recording_info.instrument",
    "documentary_stamp_tax",
]


# ---- value normalization / comparison --------------------------------------
def norm_money(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else f
    if isinstance(v, str):
        s = v.strip().replace("$", "").replace(",", "")
        if s == "" or s.lower() == "null":
            return None
        try:
            f = float(s)
            return int(f) if f.is_integer() else f
        except ValueError:
            return v.strip()
    return v


def norm_str(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        return None if s == "" or s.lower() == "null" else s
    return v


def get_leaf(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return ("__MISSING__",)  # sentinel: never equals a real value
        cur = cur[part]
    return cur


def leaf_matches(path, pred, ref):
    if isinstance(pred, tuple) and pred and pred[0] == "__MISSING__":
        return False
    field = path.split(".")[-1]
    if path in MONEY or field in {"loan_amount", "purchase_price", "documentary_stamp_tax"}:
        return norm_money(pred) == norm_money(ref)
    return norm_str(pred) == norm_str(ref)


# ---- schema conformance -----------------------------------------------------
def is_str_or_null(v):
    return v is None or isinstance(v, str)


def is_num_or_null(v):
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def schema_conforms(obj):
    if not isinstance(obj, dict) or set(obj.keys()) != TOP_KEYS:
        return False
    for k in STR_OR_NULL:
        if not is_str_or_null(obj[k]):
            return False
    for k in MONEY:
        if not is_num_or_null(obj[k]):
            return False
    for sub, keys in (("parties", PARTY_KEYS), ("dates", DATE_KEYS), ("recording_info", REC_KEYS)):
        d = obj[sub]
        if not isinstance(d, dict) or set(d.keys()) != keys:
            return False
        if not all(is_str_or_null(d[k]) for k in keys):
            return False
    return True


# ---- output parsing ---------------------------------------------------------
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text):
    """Return (parsed_or_None, cleaned_text). Strips <think> blocks and pulls
    the outermost {...} object if there is surrounding prose."""
    cleaned = THINK_RE.sub("", text).strip()
    # strip common code fences
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate), candidate
        except json.JSONDecodeError:
            return None, cleaned
    return None, cleaned


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo or local path")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter dir (optional)")
    ap.add_argument("--system-prompt", default=None,
                    help="System prompt text, or the literal 'SCHEMA' to use the "
                         "built-in schema-describing prompt")
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--output", default=None, help="per-example results JSON")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="eval only first N examples")
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    system_prompt = args.system_prompt
    if system_prompt == "SCHEMA":
        system_prompt = SCHEMA_SYSTEM_PROMPT

    examples = [json.loads(l) for l in open(args.test)]
    if args.limit:
        examples = examples[:args.limit]

    print(f"Loading model={args.model} adapter={args.adapter_path} ...", flush=True)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    sampler = make_sampler(temp=0.0)  # greedy / deterministic

    def build_prompt(user_content):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_content})
        try:
            return tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    n = len(examples)
    valid_json = 0
    schema_ok = 0
    field_hits = defaultdict(int)
    field_total = defaultdict(int)
    overall_hits = 0
    overall_total = 0
    per_example = []

    for i, ex in enumerate(examples, 1):
        user = ex["messages"][0]["content"]
        ref = json.loads(ex["messages"][1]["content"])
        prompt = build_prompt(user)
        raw = generate(model, tokenizer, prompt=prompt,
                       max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        parsed, cleaned = extract_json(raw)

        is_valid = parsed is not None
        is_schema = bool(is_valid and schema_conforms(parsed))
        valid_json += int(is_valid)
        schema_ok += int(is_schema)

        field_results = {}
        for path in LEAF_FIELDS:
            ref_val = get_leaf(ref, path)
            pred_val = get_leaf(parsed, path) if is_valid else ("__MISSING__",)
            m = leaf_matches(path, pred_val, ref_val)
            field_results[path] = m
            field_total[path] += 1
            overall_total += 1
            if m:
                field_hits[path] += 1
                overall_hits += 1

        per_example.append({
            "index": i - 1,
            "user_preview": user.split("JSON:\n\n", 1)[-1][:120],
            "valid_json": is_valid,
            "schema_conforms": is_schema,
            "field_matches": field_results,
            "n_field_matches": sum(field_results.values()),
            "reference": ref,
            "prediction": parsed,
            "raw_output": raw,
        })
        print(f"[{i}/{n}] valid={is_valid} schema={is_schema} "
              f"fields={sum(field_results.values())}/{len(LEAF_FIELDS)}", flush=True)

    # ---- summary ----
    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "system_prompt": system_prompt,
        "n_examples": n,
        "valid_json_rate": round(100 * valid_json / n, 2) if n else 0.0,
        "schema_conformance_rate": round(100 * schema_ok / n, 2) if n else 0.0,
        "field_accuracy_overall": round(100 * overall_hits / overall_total, 2) if overall_total else 0.0,
        "field_accuracy_per_field": {
            path: round(100 * field_hits[path] / field_total[path], 2)
            for path in LEAF_FIELDS
        },
    }

    out_path = args.output
    if out_path is None:
        tag = args.model.split("/")[-1]
        if args.adapter_path:
            tag += "_adapter"
        if system_prompt:
            tag += "_schemaprompt"
        out_path = str(Path(__file__).parent / "results" / f"eval_{tag}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "examples": per_example}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"model: {args.model}  adapter: {args.adapter_path}")
    print(f"examples: {n}")
    print(f"Valid-JSON rate:          {summary['valid_json_rate']:.2f}%")
    print(f"Schema-conformance rate:  {summary['schema_conformance_rate']:.2f}%")
    print(f"Field accuracy (overall): {summary['field_accuracy_overall']:.2f}%")
    print("Field accuracy per field:")
    for path in LEAF_FIELDS:
        print(f"  {path:34s} {summary['field_accuracy_per_field'][path]:6.2f}%")
    print(f"\nper-example results -> {out_path}")


if __name__ == "__main__":
    main()
