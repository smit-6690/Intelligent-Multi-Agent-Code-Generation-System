from __future__ import annotations

from .agents import DeveloperAgent, RepairAgent, ReviewerAgent, SpecAnalyzer, TesterAgent
from .backend import MockBackend, TransformersBackend
from .config import Settings
from .retrieval import EmptyRetriever, FaissRetriever
from .types import GenerationResult


class MultiAgentPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        if self.settings.model_backend == "transformers":
            backend = TransformersBackend(
                self.settings.base_model, self.settings.adapter_path
            )
        else:
            backend = MockBackend()

        if self.settings.rag_index_path and self.settings.rag_documents_path:
            retriever = FaissRetriever(
                self.settings.rag_index_path,
                self.settings.rag_documents_path,
                self.settings.embedding_model,
            )
        else:
            retriever = EmptyRetriever()

        self.analyzer = SpecAnalyzer()
        self.developer = DeveloperAgent(backend)
        self.tester = TesterAgent(self.settings.execution_timeout_seconds)
        self.reviewer = ReviewerAgent()
        self.repair = RepairAgent(backend)
        self.retriever = retriever

    def run(self, specification: str) -> GenerationResult:
        normalized = self.analyzer.analyze(specification)
        context = self.retriever.retrieve(normalized, self.settings.rag_top_k)
        code = self.developer.generate(normalized, context)
        attempts = 0

        while True:
            tests = self.tester.test(code, normalized)
            review = self.reviewer.review(code, tests)
            if all(test.passed for test in tests) or attempts >= self.settings.max_repair_attempts:
                break
            code = self.repair.repair(code, review, normalized)
            attempts += 1

        return GenerationResult(
            specification=normalized,
            code=code,
            retrieved_context=context,
            tests=tests,
            review=review,
            repair_attempts=attempts,
            metadata={"backend": self.settings.model_backend},
        )
