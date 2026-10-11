from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class GenerationResult:
    specification: str
    code: str
    retrieved_context: list[str] = field(default_factory=list)
    tests: list[TestResult] = field(default_factory=list)
    review: str = ""
    repair_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.tests) and all(test.passed for test in self.tests)
