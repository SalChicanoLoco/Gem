"""
Local Edge RAG Engine (agents/rag_agent.py)

Ingests local text files and markdown documents, extracts semantic chunks,
builds TF-IDF / keyword vector indices, and provides context retrieval for Gemma.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional
from .gemma_agent import GemmaAgent
from .spine import get_spine


class RAGAgent:
    """
    Local Edge RAG Agent for zero-cloud document retrieval.
    """

    def __init__(self, gemma_agent: Optional[GemmaAgent] = None):
        """
        Initialize RAGAgent.

        Args:
            gemma_agent: GemmaAgent instance for query answering.
        """
        self.gemma = gemma_agent or GemmaAgent()
        self.spine = get_spine()
        self._documents: Dict[str, str] = {}
        self._chunks: List[Dict[str, Any]] = []

    def ingest_text(self, doc_id: str, content: str, chunk_size: int = 300) -> int:
        """
        Ingest text content and split into searchable chunks.

        Args:
            doc_id: Unique document identifier.
            content: Document text content.
            chunk_size: Target words per chunk.

        Returns:
            Number of chunks created.
        """
        self._documents[doc_id] = content
        words = content.split()
        num_chunks = 0

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i : i + chunk_size])
            self._chunks.append({
                "doc_id": doc_id,
                "chunk_index": num_chunks,
                "text": chunk_text,
                "words": set(re.findall(r"\w+", chunk_text.lower())),
            })
            num_chunks += 1

        return num_chunks

    def ingest_file(self, filepath: str) -> int:
        """
        Ingest text file from local filesystem.

        Args:
            filepath: Path to text/markdown file.

        Returns:
            Number of chunks ingested.
        """
        if not os.path.exists(filepath):
            return 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.ingest_text(os.path.basename(filepath), content)

    def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Ingest open-source web data from a public URL or open dataset endpoint.

        Args:
            url: Public web URL or dataset endpoint.

        Returns:
            Dictionary with ingestion summary and chunk count.
        """
        try:
            import requests
            resp = requests.get(url, timeout=5.0, headers={"User-Agent": "SenaAIgent-WebIngest/1.2"})
            if resp.status_code == 200:
                # Strip basic HTML tags if present
                clean_text = re.sub(r"<[^>]+>", " ", resp.text)
                clean_text = re.sub(r"\s+", " ", clean_text).strip()
                chunks = self.ingest_text(doc_id=url, content=clean_text)
                return {
                    "success": True,
                    "url": url,
                    "bytes_downloaded": len(resp.content),
                    "chunks_created": chunks,
                    "total_documents_in_rag": len(self._documents),
                }
            return {"success": False, "error": f"HTTP status {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search ingested chunks for top matching results based on keyword overlap scoring.

        Args:
            query: Query string.
            top_k: Number of top results to return.

        Returns:
            List of matching chunks with similarity score.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words or not self._chunks:
            return []

        scored_chunks = []
        for chunk in self._chunks:
            overlap = len(query_words.intersection(chunk["words"]))
            score = overlap / (math.log(len(chunk["words"]) + 1) + 1.0)
            if overlap > 0:
                scored_chunks.append({
                    "doc_id": chunk["doc_id"],
                    "score": round(score, 3),
                    "text": chunk["text"],
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def query(self, query: str) -> Dict[str, Any]:
        """
        Perform RAG query: retrieve context and generate Gemma answer.

        Args:
            query: User query.

        Returns:
            Dictionary with retrieved context chunks and LLM generated answer.
        """
        results = self.search(query, top_k=3)
        context = "\n---\n".join([r["text"] for r in results]) if results else "No relevant local context found."

        prompt = (
            f"Use the following local context to answer the query:\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}\n"
        )

        answer = self.gemma.generate(prompt, temperature=0.3)

        return {
            "success": True,
            "query": query,
            "context_chunks_retrieved": len(results),
            "retrieved_results": results,
            "answer": answer,
        }
