from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .pipeline import MultiAgentPipeline


def load_tasks(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        return [json.loads(line) for line in file_path.read_text().splitlines() if line.strip()]
    payload = json.loads(file_path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return payload["tasks"]
    raise ValueError("Expected a JSON list, JSONL file, or an object containing `tasks`.")


def evaluate_tasks(
    pipeline: MultiAgentPipeline, tasks: Iterable[dict], limit: int | None = None
) -> dict:
    records: list[dict] = []
    for index, task in enumerate(tasks):
        if limit is not None and index >= limit:
            break
        prompt = task.get("prompt") or task.get("specification") or task.get("text")
        if not prompt:
            records.append({"index": index, "status": "skipped", "reason": "missing prompt"})
            continue
        result = pipeline.run(prompt)
        records.append(
            {
                "index": index,
                "task_id": task.get("task_id"),
                "status": "passed" if result.passed else "failed",
                "repair_attempts": result.repair_attempts,
                "review": result.review,
                "code": result.code,
            }
        )

    evaluated = [record for record in records if record["status"] != "skipped"]
    passed = sum(record["status"] == "passed" for record in evaluated)
    repaired = sum(record["repair_attempts"] > 0 for record in evaluated)
    return {
        "metric_name": "pipeline_smoke_pass_rate",
        "evaluated": len(evaluated),
        "passed": passed,
        "failed": len(evaluated) - passed,
        "repaired": repaired,
        "smoke_pass_rate": passed / len(evaluated) if evaluated else 0.0,
        "records": records,
    }
