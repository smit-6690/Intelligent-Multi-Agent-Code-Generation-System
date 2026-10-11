# Stored experiment artifacts

This directory contains small, human-readable evaluation summaries copied from the project’s Drive archive. Large model, FAISS, and dataset files are intentionally not committed to Git because they are binary artifacts and may be hundreds of megabytes.

## Included summaries

- `results/phase2_complete_rtx4090.json`: Phase 2 run summary. It records 100,000 RAG samples, 127,264 chunks, 100 evaluated samples, 68 passed, 32 repaired, and 0 failed.
- `results/analysis_report_100_samples.json`: Multi-model and strategy analysis report. Its `avg_quality_score` values must not be described as HumanEval `pass@1` without a separate benchmark calculation.

## Large artifacts to download locally

The Drive archive also contains:

- A CodeSearchNet Python FAISS index and JSONL documents.
- Phase 3 PEFT/QLoRA adapters and checkpoints.
- Formatted training and evaluation datasets.

Set `RAG_INDEX_PATH`, `RAG_DOCUMENTS_PATH`, and `ADAPTER_PATH` in `.env` after downloading those artifacts. The pipeline can run in `MODEL_BACKEND=mock` mode without them for local smoke tests.
