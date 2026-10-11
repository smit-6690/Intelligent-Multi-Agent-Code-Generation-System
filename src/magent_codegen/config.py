from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Optional for the dependency-free smoke-test mode.
    def load_dotenv(*args, **kwargs):
        del args, kwargs


@dataclass(frozen=True)
class Settings:
    model_backend: str = "mock"
    base_model: str = "mistralai/Mistral-7B-v0.1"
    adapter_path: str | None = None
    rag_index_path: str | None = None
    rag_documents_path: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4
    max_repair_attempts: int = 2
    execution_timeout_seconds: int = 5

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file, override=False)
        return cls(
            model_backend=os.getenv("MODEL_BACKEND", "mock"),
            base_model=os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1"),
            adapter_path=os.getenv("ADAPTER_PATH") or None,
            rag_index_path=os.getenv("RAG_INDEX_PATH") or None,
            rag_documents_path=os.getenv("RAG_DOCUMENTS_PATH") or None,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            rag_top_k=max(1, int(os.getenv("RAG_TOP_K", "4"))),
            max_repair_attempts=max(0, int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))),
            execution_timeout_seconds=max(
                1, int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "5"))
            ),
        )
