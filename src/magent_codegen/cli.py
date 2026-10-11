from __future__ import annotations

import argparse
import json

from .config import Settings
from .pipeline import MultiAgentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent code pipeline.")
    parser.add_argument("specification", help="Natural-language coding task")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    args = parser.parse_args()

    result = MultiAgentPipeline(Settings.from_env()).run(args.specification)
    payload = {
        "code": result.code,
        "passed": result.passed,
        "repair_attempts": result.repair_attempts,
        "review": result.review,
        "backend": result.metadata.get("backend"),
    }
    print(json.dumps(payload, indent=2) if args.json else result.code)


if __name__ == "__main__":
    main()
