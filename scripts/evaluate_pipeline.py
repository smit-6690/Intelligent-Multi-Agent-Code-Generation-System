from __future__ import annotations

import argparse
import json
from pathlib import Path

from magent_codegen.config import Settings
from magent_codegen.evaluation import evaluate_tasks, load_tasks
from magent_codegen.pipeline import MultiAgentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on JSON or JSONL tasks.")
    parser.add_argument("--input", required=True, help="JSON or JSONL task file")
    parser.add_argument("--output", required=True, help="Output JSON report")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = evaluate_tasks(
        MultiAgentPipeline(Settings.from_env()),
        load_tasks(args.input),
        args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
