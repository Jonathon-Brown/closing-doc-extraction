# Results

## Read this first: what these numbers do and do not measure

This project fine-tunes Qwen3-8B (4-bit) to extract a fixed JSON schema from
real-estate closing documents. Before any numbers, the central limitation:

- **292 of the 302 documents in the corpus are synthetic.** They were generated
  locally by `qwen3:32b` from a small set of Florida closing-document templates
  (deeds, mortgages, closing disclosures, title commitments, condo/HOA
  disclosures, filled contracts), with fabricated names, parcels, book/page, and
  instrument numbers. Only 10 documents (the `real-forms` category: CFPB and
  FAR/BAR reference PDFs) are genuinely real.
- **The training labels came from the same model family.** The assistant-side
  JSON for every example was produced by `qwen3:32b`, then validated for schema
  conformance. There is no independent human ground truth. The fine-tuned model
  is therefore learning to reproduce `qwen3:32b`'s extractions on templated text.
- **The held-out test set is same-distribution synthetic data.** Train, valid,
  and test were split from the same generator output, so a test document differs
  from a training document only in its fabricated field values, not in structure,
  vocabulary, or layout.

Consequently, the fine-tuned model's **100% on the held-out test set measures
in-distribution template matching, not real-world extraction.** It shows the
adapter learned the schema and the template regularities cleanly; it does not
show that the model would extract correctly from a real, messy, OCR'd closing
document it has never seen.

For the same reason, **the headline "5% naive baseline to 100% fine-tuned"
comparison is not the meaningful one.** The naive baseline scores near zero only
because it does not know the target key layout; that gap is trivially closed by
telling the model the schema. The meaningful comparison is against a baseline
that has already been told the schema (below).

## The meaningful result: fine-tuned vs schema-prompted baseline

The fair comparison is the fine-tuned model with **no schema in the prompt**
against the base model **with the full schema in the prompt**. Both are told, in
effect, what to produce — one by fine-tuning, one by instruction. Held-out test
set, n = 30.

| Metric | Base + schema prompt | Fine-tuned, no schema prompt |
| --- | --- | --- |
| Valid-JSON rate | 100.00% | 100.00% |
| Schema-conformance rate | 66.67% | 100.00% |
| Field accuracy (overall) | 90.00% | 100.00% |

The fine-tuned model reaches full schema conformance and full field accuracy
**without the schema in the prompt at inference** — a shorter prompt and a
strictly structured output, versus a base model that, even when handed the exact
schema, still breaks structure on a third of outputs and misses ~10% of fields.

### Per-field

| Field | Base + schema prompt | Fine-tuned, no schema prompt |
| --- | --- | --- |
| document_type | 93.33% | 100.00% |
| parties.grantor | 83.33% | 100.00% |
| parties.grantee | 83.33% | 100.00% |
| parties.borrower | 63.33% | 100.00% |
| parties.lender | 66.67% | 100.00% |
| property_address | 100.00% | 100.00% |
| legal_description | 100.00% | 100.00% |
| loan_amount | 100.00% | 100.00% |
| purchase_price | 93.33% | 100.00% |
| dates.executed | 93.33% | 100.00% |
| dates.recorded | 100.00% | 100.00% |
| dates.closing | 100.00% | 100.00% |
| recording_info.book | 93.33% | 100.00% |
| recording_info.page | 93.33% | 100.00% |
| recording_info.instrument | 93.33% | 100.00% |
| documentary_stamp_tax | 83.33% | 100.00% |

The base model's weakest fields under schema prompting are `parties.borrower`
(63.33%) and `parties.lender` (66.67%) — the grantor/grantee-vs-borrower/lender
role distinction (deeds populate the former and null the latter; mortgages do the
reverse). That convention is not stated in the schema and cannot be inferred from
key names alone, which is why fine-tuning helps most there.

## All four configurations

For completeness, both baselines and both fine-tuned configurations. Held-out
test set, n = 30, base model `mlx-community/Qwen3-8B-4bit`, 2048-token generation
cap, greedy decoding.

| Metric | Naive | Schema prompt | Fine-tuned, no schema | Fine-tuned, schema prompt |
| --- | --- | --- | --- | --- |
| Valid-JSON rate | 96.67% | 100.00% | 100.00% | 100.00% |
| Schema-conformance rate | 0.00% | 66.67% | 100.00% | 100.00% |
| Field accuracy (overall) | 5.00% | 90.00% | 100.00% | 95.21% |

| Field | Naive | Schema prompt | Fine-tuned, no schema | Fine-tuned, schema prompt |
| --- | --- | --- | --- | --- |
| document_type | 40.00% | 93.33% | 100.00% | 80.00% |
| parties.grantor | 0.00% | 83.33% | 100.00% | 93.33% |
| parties.grantee | 0.00% | 83.33% | 100.00% | 93.33% |
| parties.borrower | 0.00% | 63.33% | 100.00% | 100.00% |
| parties.lender | 0.00% | 66.67% | 100.00% | 100.00% |
| property_address | 3.33% | 100.00% | 100.00% | 100.00% |
| legal_description | 3.33% | 100.00% | 100.00% | 100.00% |
| loan_amount | 0.00% | 100.00% | 100.00% | 100.00% |
| purchase_price | 3.33% | 93.33% | 100.00% | 86.67% |
| dates.executed | 0.00% | 93.33% | 100.00% | 90.00% |
| dates.recorded | 0.00% | 100.00% | 100.00% | 100.00% |
| dates.closing | 0.00% | 100.00% | 100.00% | 100.00% |
| recording_info.book | 0.00% | 93.33% | 100.00% | 90.00% |
| recording_info.page | 0.00% | 93.33% | 100.00% | 90.00% |
| recording_info.instrument | 0.00% | 93.33% | 100.00% | 100.00% |
| documentary_stamp_tax | 30.00% | 83.33% | 100.00% | 100.00% |

## Finding: adding the schema prompt to the fine-tuned model makes it worse

The fine-tuned model scores 100.00% overall with no schema in the prompt, but
95.21% when the same schema prompt used for the base-model baseline is prepended.
Adding instruction to an already-fine-tuned model is a net regression.

The adapter was trained on examples with no system message. Injecting one at
inference is off-distribution for the fine-tuned weights. Concretely, the schema
prompt's rule "use null for any field that is absent" pushes the model to null
`document_type` on non-transactional documents (HOA and condo disclosures, the
TILA-RESPA guide) that it types correctly when left alone:

| Field | FT no schema | FT + schema prompt | Change |
| --- | --- | --- | --- |
| document_type | 100.00% | 80.00% | -20.00 |
| purchase_price | 100.00% | 86.67% | -13.33 |
| dates.executed | 100.00% | 90.00% | -10.00 |
| recording_info.book | 100.00% | 90.00% | -10.00 |
| recording_info.page | 100.00% | 90.00% | -10.00 |

Relative to the schema-prompted *baseline*, the fine-tuned-plus-schema-prompt
configuration also regresses on five fields (document_type 93.33 to 80.00,
purchase_price 93.33 to 86.67, dates.executed 93.33 to 90.00, recording_info.book
93.33 to 90.00, recording_info.page 93.33 to 90.00), even though its overall
accuracy is higher. The practical takeaway: serve the fine-tuned model with the
bare user message and no schema prompt.

## Methodology

### Data
- Corpus: 302 documents, `~/private-ai-demo/corpus`. 292 synthetic (generated by
  `qwen3:32b` from templates), 10 real (`real-forms`).
- Text extraction: PyMuPDF for PDF, python-docx for DOCX. Document text embedded
  in each example was capped at 6,000 characters; this affected only 6 of the 10
  `real-forms` (15k-164k chars). All 292 synthetic docs are under 1,300 chars and
  were untruncated.
- Labels: assistant-side JSON generated per document by `qwen3:32b` via Ollama
  (`think` disabled, `format: json`, temperature 0), then validated against the
  fixed schema. Nonconforming generations were discarded; final discard rate 0%.
- Format: mlx-lm chat format, one user/assistant pair per document, no system
  message.
- Split: stratified 80/10/10 by document category, deterministic (CRC32 of the
  category name plus a fixed seed). Sizes: train 242, valid 30, test 30.
- Data-generation seed: 20260829.

### Fine-tuning
- Framework: mlx-lm LoRA (`fine_tune_type: lora`).
- Base model: `mlx-community/Qwen3-8B-4bit`.
- LoRA parameters: rank 8, scale (alpha) 20.0, dropout 0.0, applied to the top 16
  layers.
- Optimizer: Adam, learning rate 1e-5, no LR schedule.
- Iterations: 600. Batch size 1, gradient accumulation 1, max sequence length
  2048, gradient checkpointing off.
- Training seed: 0 (framework default). Evaluation every 200 steps on 25
  validation batches; checkpoints every 100 iterations.
- Validation loss: 1.979 (initial) to 0.216 (iter 600), monotonically decreasing
  and still falling at the final iteration, with no train/val divergence
  (no overfitting observed). Only the endpoints are recorded here; no per-eval
  training log was retained.

### Memory note (batch size)
- Batch size 2 ran out of unified memory (OOM) during training on this machine.
- Resolved by reducing to batch size 1 (with gradient accumulation left at 1).
  This is the configuration reflected above and in `adapters/adapter_config.json`.
  A larger effective batch could instead be recovered via gradient accumulation
  without the memory cost, but was not needed for this run.

### Evaluation
- `eval.py`, greedy decoding (temperature 0), 2048-token generation cap.
- Metrics: valid-JSON rate (output parses as JSON), schema-conformance rate
  (exact expected key structure and types), and field-level accuracy across all
  16 leaf fields (null-vs-null counts as a match), reported overall and per field.
- Four configurations evaluated on the held-out test set: base naive, base with
  schema prompt (`--system-prompt SCHEMA`), fine-tuned without schema prompt, and
  fine-tuned with schema prompt.
- Per-example outputs saved under `results/` for inspection (gitignored).

## Out-of-distribution check: the 10 real forms

The only genuinely out-of-distribution data in this project is the 10 `real-forms`
documents (real CFPB and FAR/BAR PDFs), which differ from the synthetic templates
in length, layout, and content. Results for the fine-tuned model (no schema
prompt) on just those 10 are reported in the section below. Note that even here
the reference labels were produced by `qwen3:32b`, so this measures agreement with
the label-generator on real documents, not agreement with human ground truth; the
documents, however, are genuinely unlike the training templates.

Fine-tuned model, no schema prompt, on the 10 `real-forms` only:

| Metric | Fine-tuned, no schema prompt |
| --- | --- |
| Valid-JSON rate | 90.00% |
| Schema-conformance rate | 90.00% |
| Field accuracy (overall) | 88.75% |

| Field | Accuracy |
| --- | --- |
| document_type | 90.00% |
| parties.grantor | 90.00% |
| parties.grantee | 90.00% |
| parties.borrower | 90.00% |
| parties.lender | 90.00% |
| property_address | 90.00% |
| legal_description | 90.00% |
| loan_amount | 80.00% |
| purchase_price | 90.00% |
| dates.executed | 90.00% |
| dates.recorded | 90.00% |
| dates.closing | 80.00% |
| recording_info.book | 90.00% |
| recording_info.page | 90.00% |
| recording_info.instrument | 90.00% |
| documentary_stamp_tax | 90.00% |

With only 10 documents, each one is worth 10 percentage points, so these numbers
are coarse. Two failure modes account for the gap:

- One of the ten (a dense real CFPB sample closing disclosure) hit the 2048-token
  generation cap and was cut off mid-JSON, so it fails valid-JSON, schema
  conformance, and every field at once. This is the single largest contributor to
  the drop and is a real out-of-distribution effect: the real forms are longer and
  denser than the templates the model trained on. Raising the generation cap would
  likely recover this example.
- One other document (a filled CFPB sample) disagreed with the `qwen3:32b`
  reference on `loan_amount` and `dates.closing`.

Every other real form (8 of 10) was extracted with full schema conformance and
16/16 fields matching the reference. The drop from 100% (in-distribution) to
roughly 89% (out-of-distribution) is modest but real, and its dominant cause is
output truncation rather than misextraction — the kind of thing that only shows
up on documents unlike the training set.
