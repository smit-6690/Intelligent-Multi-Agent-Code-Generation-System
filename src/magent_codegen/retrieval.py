from __future__ import annotations

import json
from pathlib import Path


class EmptyRetriever:
    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        del query, top_k
        return []


class FaissRetriever:
    def __init__(self, index_path: str, documents_path: str, embedding_model: str):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install RAG dependencies with `pip install -e '.[ml]'`."
            ) from exc

        self._faiss = faiss
        self._model = SentenceTransformer(embedding_model)
        self._index = faiss.read_index(str(index_path))
        self._documents = self._load_documents(Path(documents_path))

    @staticmethod
    def _load_documents(path: Path) -> list[str]:
        documents: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if isinstance(record, str):
                    documents.append(record)
                else:
                    documents.append(
                        record.get("text")
                        or record.get("code")
                        or record.get("content")
                        or json.dumps(record)
                    )
        return documents

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        vector = self._model.encode([query], normalize_embeddings=True)
        _, indices = self._index.search(vector, top_k)
        return [
            self._documents[index]
            for index in indices[0]
            if 0 <= index < len(self._documents)
        ]
