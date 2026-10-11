# Intelligent Multi-Agent Code Generation System

A Python-based multi-agent code-generation pipeline that combines specification analysis, retrieval-augmented generation, automated testing, code review, and iterative repair.

The system improves the reliability of generated code by validating outputs before returning them to the user.

## Overview

The pipeline contains five logical agents:

1. Specification Analyzer
2. Developer Agent
3. Tester Agent
4. Reviewer Agent
5. Repair Agent

```text
User Specification
        |
        v
Specification Analyzer
        |
        v
RAG Retriever ------> Code Context
        |
        v
Developer Agent
        |
        v
Tester Agent
        |
        v
Reviewer Agent
        |
        v
Repair Agent
        |
        v
Validated Generated Code
```

## Key Features

- Five-stage multi-agent orchestration
- Retrieval-augmented code generation
- FAISS-based code retrieval
- CodeSearchNet Python corpus support
- Transformers and PEFT adapter integration
- Optional QLoRA-compatible model workflow
- Automated syntax and smoke validation
- Iterative code repair
- Command-line interface
- HTTP API
- Reproducible evaluation scripts
- Stored experiment summaries
- Dependency-light mock mode for local development

## Technology Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- PEFT
- FAISS
- Sentence Transformers
- Hugging Face Datasets
- pytest
- REST API
- JSON and JSONL evaluation artifacts

## Recorded Experiment Metrics

The stored experiment artifacts contain the following results:

| Metric | Result |
|---|---:|
| RAG samples | 100,000 |
| Generated/indexed code chunks | 127,264 |
| Evaluated examples | 100 |
| Initial passing examples | 68 |
| Repaired examples | 32 |
| Final failed examples | 0 |
| Base experiment hardware | NVIDIA RTX 4090 |
| Adapter type | PEFT LoRA |
| Adapter rank | 32 |
| LoRA alpha | 64 |

## Historical Experiment Metrics

Previous experiment notes reported the following comparisons:

| Metric | Reported Result |
|---|---:|
| Initial functional performance | 68% |
| Reported post-processing quality score | 93.97% |
| Reported GPU memory usage before optimization | 13.7 GB |
| Reported GPU memory usage after optimization | 3.6 GB |
| Reported GPU memory reduction | Approximately 74% |
| Reported throughput improvement | Approximately 3.8× |
| Reported service availability | 99.7% |


The memory, throughput, and availability values represent historical experiment notes and depend on the model, hardware, workload, dataset, and runtime configuration.

## Repository Structure

```text
.
├── 01_project-testing (1).ipynb
├── 02_phase-2_pipeline_RTX4090 (1).ipynb
├── 03_phase-3_INDUSTRY_STANDARD.ipynb
├── artifacts/
│   ├── adapters/
│   │   └── checkpoint-51/
│   │       └── adapter_config.json
│   ├── rag/
│   │   └── manifest.json
│   ├── results/
│   │   ├── analysis_report_100_samples.json
│   │   └── phase2_complete_rtx4090.json
│   ├── artifact_manifest.json
│   └── README.md
├── scripts/
│   ├── evaluate_pipeline.py
│   └── validate_results.py
├── src/
│   └── magent_codegen/
│       ├── __init__.py
│       ├── agents.py
│       ├── backend.py
│       ├── cli.py
│       ├── config.py
│       ├── evaluate_cli.py
│       ├── pipeline.py
│       ├── retrieval.py
│       ├── service.py
│       └── types.py
├── tests/
│   ├── test_evaluation.py
│   ├── test_pipeline.py
│   └── test_service.py
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Create the environment file:

```bash
cp .env.example .env
```

## Local Mock Mode

Mock mode runs without a GPU, model download, FAISS index, or large dataset.

Configure `.env`:

```env
MODEL_BACKEND=mock
ADAPTER_PATH=
RAG_INDEX_PATH=
RAG_DOCUMENTS_PATH=
MAX_REPAIR_ATTEMPTS=2
EXECUTION_TIMEOUT_SECONDS=5
```

Run a code-generation request:

```bash
PYTHONPATH=src python -m magent_codegen.cli \
  "Return all prime numbers up to n." \
  --json
```

Example output:

```json
{
  "code": "def solve(n):\n    ...",
  "passed": true,
  "repair_attempts": 0,
  "review": "The code passes syntax and smoke validation and defines a callable function.",
  "backend": "mock"
}
```

## Validate Stored Results

Run the artifact validation script:

```bash
PYTHONPATH=src python scripts/validate_results.py
```

Example output:

```text
Phase 2 stored result
  RAG samples: 100000
  RAG chunks:  127264
  Evaluated:   100
  Passed:      68
  Repaired:    32
  Failed:      0
```

## Run the Test Suite

```bash
pytest
```

Compile the source files:

```bash
PYTHONPATH=src python -m compileall -q src scripts tests
```

## Run the HTTP API

Start the service:

```bash
PYTHONPATH=src MODEL_BACKEND=mock \
python -m magent_codegen.service
```

The service listens on:

```text
http://127.0.0.1:8080
```

Check service health:

```bash
curl http://127.0.0.1:8080/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "magent-codegen"
}
```

Generate code through the API:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "specification": "Return all prime numbers up to n."
  }'
```

## Evaluation

Create a task file:

```bash
mkdir -p /tmp/magent-tasks
```

```bash
cat > /tmp/magent-tasks/tasks.json <<'EOF'
[
  {
    "id": "prime-numbers",
    "specification": "Return all prime numbers up to n."
  },
  {
    "id": "reverse-string",
    "specification": "Write a function that reverses a string."
  }
]
EOF
```

Run the evaluation:

```bash
PYTHONPATH=src python -m magent_codegen.evaluate_cli \
  --input /tmp/magent-tasks/tasks.json \
  --output artifacts/results/local_evaluation.json
```

The evaluation report uses the metric:

```text
pipeline_smoke_pass_rate
```

This metric represents local syntax and smoke validation. It is not a replacement for an official HumanEval `pass@1` calculation.

## Using Transformers and PEFT

Full model inference requires:

- A compatible machine with sufficient memory
- PyTorch
- Transformers
- PEFT
- Accelerate
- The base model
- The saved adapter
- The FAISS index
- The document collection
- Optional Sentence Transformers embeddings

Install the machine-learning dependencies:

```bash
python -m pip install -e ".[ml]"
```

Configure `.env`:

```env
MODEL_BACKEND=transformers
BASE_MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
ADAPTER_PATH=/absolute/path/to/checkpoint-51
RAG_INDEX_PATH=/absolute/path/to/codesearchnet_python.faiss
RAG_DOCUMENTS_PATH=/absolute/path/to/documents.jsonl
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=4
MAX_REPAIR_ATTEMPTS=2
EXECUTION_TIMEOUT_SECONDS=5
```

Run the model-backed pipeline:

```bash
PYTHONPATH=src python -m magent_codegen.cli \
  "Return all prime numbers up to n." \
  --json
```

The base model and PEFT adapter must be compatible. The stored adapter configuration targets:

```text
deepseek-ai/deepseek-coder-6.7b-instruct
```
