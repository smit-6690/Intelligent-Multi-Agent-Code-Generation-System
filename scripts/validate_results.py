from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    phase2 = json.loads((root / "artifacts/results/phase2_complete_rtx4090.json").read_text())
    required = {"rag_samples", "rag_chunks", "samples", "passed", "repaired", "failed"}
    missing = required - phase2.keys()
    if missing:
        raise SystemExit(f"Missing fields in phase 2 result: {sorted(missing)}")
    print("Phase 2 stored result")
    print(f"  RAG samples: {phase2['rag_samples']}")
    print(f"  RAG chunks:  {phase2['rag_chunks']}")
    print(f"  Evaluated:   {phase2['samples']}")
    print(f"  Passed:      {phase2['passed']}")
    print(f"  Repaired:    {phase2['repaired']}")
    print(f"  Failed:      {phase2['failed']}")

    manifest = json.loads((root / "artifacts/artifact_manifest.json").read_text())
    index = root / manifest["rag"]["index"]["local_path"]
    documents = root / manifest["rag"]["documents"]["local_path"]
    if index.exists():
        actual_size = index.stat().st_size
        expected_size = manifest["rag"]["index"]["expected_bytes"]
        if actual_size != expected_size:
            raise SystemExit(f"RAG index size mismatch: {actual_size} != {expected_size}")
        print(f"  RAG index:  verified ({actual_size} bytes)")
    else:
        print("  RAG index:  not downloaded")
    if documents.exists():
        line_count = sum(1 for _ in documents.open(encoding="utf-8"))
        expected_lines = manifest["rag"]["documents"]["expected_lines"]
        if line_count != expected_lines:
            raise SystemExit(f"Document count mismatch: {line_count} != {expected_lines}")
        print(f"  Documents:   verified ({line_count} lines)")
    else:
        print("  Documents:   not downloaded")

    adapter = root / manifest["adapter"]["local_path"]
    if adapter.exists():
        actual_size = adapter.stat().st_size
        expected_size = manifest["adapter"]["expected_bytes"]
        if actual_size != expected_size:
            raise SystemExit(f"Adapter size mismatch: {actual_size} != {expected_size}")
        print(f"  PEFT adapter: verified ({actual_size} bytes)")
    else:
        print("  PEFT adapter: not downloaded")


if __name__ == "__main__":
    main()
