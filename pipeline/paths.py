"""Where the pipeline reads and writes.

Everything lands under one working directory: ./work at the repo root by
default, or wherever CDE_WORKDIR points. It holds the corpus, the extracted
text, the labeled examples and the train/valid/test splits, and it is
gitignored (the corpus is generated, and the real forms are third-party PDFs).
"""
import os
from pathlib import Path

WORKDIR = Path(os.environ.get("CDE_WORKDIR", Path(__file__).resolve().parent.parent / "work"))
CORPUS = WORKDIR / "corpus"
FINETUNE = WORKDIR / "finetune"
