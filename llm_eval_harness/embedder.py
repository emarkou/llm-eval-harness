"""Embedder using Voyage AI (Anthropic's embedding service) + in-memory vector store.

Requires VOYAGE_API_KEY environment variable.
Default model: voyage-3
"""
from __future__ import annotations

import math
import os

import numpy as np
import voyageai

from llm_eval_harness.models import Chunk

_DEFAULT_MODEL = "voyage-3"
_BATCH_SIZE = 128  # Voyage API limit per request


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Embedder:
    """Calls the Voyage AI embedding API to embed text and attach results to Chunks."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOYAGE_API_KEY environment variable is not set. "
                "Get a key at https://dash.voyageai.com and run: export VOYAGE_API_KEY=<your-key>"
            )
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of strings, batching as needed."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            result = self._client.embed(batch, model=self._model)
            embeddings.extend(result.embeddings)
        return embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks in-place and return the same list."""
        texts = [c.text for c in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks


class VectorStore:
    """Simple in-memory vector store backed by numpy cosine similarity.

    Designed to be a drop-in that can be replaced by a real vector DB:
    subclass or duck-type the ``add`` / ``retrieve`` interface.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (N, D) once built

    def add(self, chunks: list[Chunk]) -> None:
        """Add pre-embedded chunks. Rebuilds the index matrix."""
        if any(c.embedding is None for c in chunks):
            raise ValueError("All chunks must have embeddings before adding to VectorStore.")
        self._chunks.extend(chunks)
        self._matrix = np.array([c.embedding for c in self._chunks], dtype=np.float32)

    def retrieve(self, query_embedding: list[float], top_k: int = 3) -> list[Chunk]:
        """Return the top-k most similar chunks to query_embedding."""
        if self._matrix is None or len(self._chunks) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        # Vectorised cosine similarity
        norms = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        scores = self._matrix @ q / norms
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._chunks[i] for i in top_indices]
