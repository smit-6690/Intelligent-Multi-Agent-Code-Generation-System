# Intelligent Multi-Agent Code Generation System

A reproducible Python pipeline for retrieval-augmented code generation, automated validation, review, and repair. The project makes the stored notebook experiments usable through a small package and command-line interface.

## What is included

- Five logical stages: specification analysis, generation, testing, review, and repair.
- Optional FAISS retrieval over the saved CodeSearchNet Python index.
- Optional Hugging Face Transformers + PEFT adapter backend.
- Dependency-light mock mode for local smoke tests.
- Stored experiment summaries under `artifacts/results/`.
- Original notebooks retained for experiment provenance.

## Verified stored results

The archived Phase 2 summary records:

- 100,000 indexed RAG samples
- 127,264 generated chunks
- 100 evaluated samples
- 68 passed on the initial evaluation
- 32 repaired
- 0 failed after the recorded repair workflow

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

cp .env.example .env
PYTHONPATH=src python -m magent_codegen.cli \
  "Return all prime numbers up to n." --json
```

Validate the stored result:

```bash
PYTHONPATH=src python scripts/validate_results.py
```

Evaluate a JSON or JSONL task set:

```bash
PYTHONPATH=src python -m magent_codegen.evaluate_cli \
  --input path/to/tasks.json \
  --output artifacts/results/local_evaluation.json
```

The generated report uses the explicit metric name `pipeline_smoke_pass_rate`. It is not a substitute for HumanEval `pass@1`; a benchmark harness must execute each task's official tests before reporting that metric.

## Run the local API

Start the dependency-light API in mock mode:

```bash
PYTHONPATH=src MODEL_BACKEND=mock python -m magent_codegen.service
```

Check health:

```bash
curl http://127.0.0.1:8080/health
```

Generate code:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"specification":"Return all prime numbers up to n."}'
```

Run tests after installing the development extra:

```bash
pytest
```

## Using the saved model and RAG artifacts

The Drive archive contains the FAISS index, JSONL documents, PEFT adapter, checkpoints, and formatted datasets. These large files are intentionally kept outside Git. After downloading them locally, set:

```env
MODEL_BACKEND=transformers
BASE_MODEL=mistralai/Mistral-7B-v0.1
ADAPTER_PATH=/absolute/path/to/adapter-directory
RAG_INDEX_PATH=/absolute/path/to/codesearchnet_python.faiss
RAG_DOCUMENTS_PATH=/absolute/path/to/documents.jsonl
```

Install the optional dependencies:

```bash
python -m pip install -e ".[ml]"
```

Then run the same CLI command. The backend loads the base model lazily and applies the PEFT adapter when `ADAPTER_PATH` is configured.

The adapter stored in the archive targets `deepseek-ai/deepseek-coder-6.7b-instruct`. The Phase 2 summary is a separate Mistral-based experiment, so those model results must not be combined without recording the model and configuration for each run.

## Repository layout

```text
src/magent_codegen/
  agents.py       # generation, testing, review, and repair stages
  backend.py      # mock and Transformers/PEFT backends
  config.py       # environment-backed settings
  pipeline.py     # orchestration entry point
  retrieval.py    # optional FAISS retriever
  types.py        # result models
  cli.py          # command-line interface
  evaluate_cli.py # evaluation command
  service.py      # standard-library HTTP API
artifacts/
  results/        # stored JSON summaries
  rag/            # small RAG manifest; large files stay local
scripts/
  validate_results.py
tests/
01_project-testing (1).ipynb
02_phase-2_pipeline_RTX4090 (1).ipynb
03_phase-3_INDUSTRY_STANDARD.ipynb
```

## Reproducibility notes

The notebooks represent the original GPU experiments. The package is the reproducible integration layer built around those artifacts. Full model inference requires the base model, compatible CUDA/PyTorch dependencies, the saved adapter, and the RAG files. The mock backend is provided for CI and local structural validation without a GPU.


