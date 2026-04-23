"""RAG runner: embeds a question, retrieves context chunks, calls the target LLM."""
from __future__ import annotations

from llm_eval_harness.embedder import Embedder, VectorStore
from llm_eval_harness.models import Chunk
from llm_eval_harness.runners.base import BaseRunner

_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the information in \
the provided context. If the answer is not present in the context, say exactly: \
"I don't have information about that in the provided documentation."

Context:
{context}

Question: {question}
Answer:"""


class RAGRunner:
    """Orchestrates retrieval and generation for a single question."""

    def __init__(
        self,
        runner: BaseRunner,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 3,
    ) -> None:
        self._runner = runner
        self._store = vector_store
        self._embedder = embedder
        self._top_k = top_k

    def run(self, question: str) -> tuple[str, list[Chunk]]:
        """Return (answer, retrieved_chunks).

        Embeds the question, fetches top-k chunks, builds a context prompt,
        and calls the target LLM.
        """
        query_embedding = self._embedder.embed_texts([question])[0]
        retrieved = self._store.retrieve(query_embedding, top_k=self._top_k)

        context = "\n\n---\n\n".join(
            f"[{c.doc_name} / chunk {c.chunk_id}]\n{c.text}" for c in retrieved
        )
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self._runner.run(prompt)
        return answer, retrieved
