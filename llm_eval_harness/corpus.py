"""Document corpus loader with overlapping chunk splitting.

Recommended documents to load for the llm-eval-harness project itself:
  - README.md        — project overview, CLI usage, input formats
  - examples/        — sample suites and prompt templates
  - llm_eval_harness/**/*.py — source docstrings (strip .py for prose-only eval)
"""
from __future__ import annotations

from pathlib import Path

from llm_eval_harness.models import Chunk


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks measured in whitespace-delimited tokens."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - chunk_overlap
    return chunks


def load_corpus(
    directory: Path,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[Chunk]:
    """Load all .txt and .md files from *directory* and split into chunks.

    Args:
        directory: Path to the directory containing documents.
        chunk_size: Maximum chunk size in (approximate) tokens.
        chunk_overlap: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of Chunk objects with ``embedding=None``.
    """
    if not directory.is_dir():
        raise ValueError(f"Corpus directory does not exist: {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.suffix not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_name = path.name
        for idx, chunk_text in enumerate(_chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_name}::{idx}",
                    doc_name=doc_name,
                    text=chunk_text,
                )
            )

    return chunks
