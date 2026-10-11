from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .backend import GenerationBackend
from .types import TestResult


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


class SpecAnalyzer:
    def analyze(self, specification: str) -> str:
        return specification.strip()


class DeveloperAgent:
    def __init__(self, backend: GenerationBackend):
        self.backend = backend

    def generate(self, specification: str, context: list[str]) -> str:
        context_block = "\n\n".join(context)
        prompt = (
            "Generate only valid Python code for the following task. "
            "Define a function named solve unless the task specifies another name.\n\n"
            f"Task:\n{specification}\n\n"
            f"Retrieved examples:\n{context_block}\n"
        )
        return extract_python_code(self.backend.generate(prompt))


class TesterAgent:
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

    def test(self, code: str, specification: str) -> list[TestResult]:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return [TestResult(False, error=f"SyntaxError: {exc}")]

        # Smoke-test the generated module without making assumptions about its API.
        test_script = (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('generated', 'generated.py')\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "assert hasattr(module, 'solve'), 'generated code must define solve'\n"
        )
        del specification
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generated.py").write_text(code, encoding="utf-8")
            (root / "test_generated.py").write_text(test_script, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, "test_generated.py"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return [TestResult(False, error="execution timed out")]
        return [
            TestResult(
                result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                error=None if result.returncode == 0 else "smoke test failed",
            )
        ]


class ReviewerAgent:
    def review(self, code: str, tests: list[TestResult]) -> str:
        if not tests or not all(test.passed for test in tests):
            return "The generated code needs repair because the smoke test did not pass."
        try:
            tree = ast.parse(code)
            has_function = any(isinstance(node, ast.FunctionDef) for node in tree.body)
        except SyntaxError:
            has_function = False
        return (
            "The code passes syntax and smoke validation and defines a callable function."
            if has_function
            else "The code should expose a top-level function for reliable integration."
        )


class RepairAgent:
    def __init__(self, backend: GenerationBackend):
        self.backend = backend

    def repair(self, code: str, review: str, specification: str) -> str:
        prompt = (
            "Repair the following Python solution. Return only code with a top-level "
            "solve function.\n\n"
            f"Task:\n{specification}\n\nCurrent code:\n{code}\n\nReview:\n{review}"
        )
        return extract_python_code(self.backend.generate(prompt))
