# closing-doc-extraction

LoRA fine-tuning of **Qwen3-8B** for structured field extraction from real-estate
closing documents, evaluated against a base-model baseline.

## Goal

Given the raw text of a closing document (deed, mortgage, closing disclosure,
title commitment, condo/HOA disclosure, or purchase contract), extract a fixed
JSON schema of key fields:

- `document_type`
- `parties` — `grantor`, `grantee`, `borrower`, `lender`
- `property_address`
- `legal_description`
- `loan_amount`, `purchase_price`, `documentary_stamp_tax`
- `dates` — `executed`, `recorded`, `closing`
- `recording_info` — `book`, `page`, `instrument`

Fields absent from a document are `null`.

## Approach

1. **Dataset** — one chat-format example per document (mlx-lm `messages` schema),
   split 80/10/10 into train/valid/test.
2. **Baseline** — evaluate base Qwen3-8B zero-shot on the test split.
3. **Fine-tune** — train a LoRA adapter on the train split.
4. **Compare** — measure the adapter against the baseline on held-out test data
   (schema validity, field-level accuracy).

## Status

Scaffolding. Dataset generation and training scripts to follow.

## License

[MIT](LICENSE) © 2026 Jonathon Brown
