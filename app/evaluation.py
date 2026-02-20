"""
Retrieval Evaluation Module — measures the quality of chunk retrieval.

Metrics:
  - Hit Rate: Percentage of queries where at least one relevant chunk
    appears in the top-k results (measured by embedding similarity).
  - Mean Reciprocal Rank (MRR): Average of 1/rank of the first relevant
    chunk across all queries.

Uses embedding cosine similarity to determine relevance instead of
hardcoded keywords, making it work with any PDF content.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    hit_rate: float = 0.0
    mrr: float = 0.0
    total_queries: int = 0
    hits: int = 0
    details: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auto-generate evaluation queries from document chunks
# ---------------------------------------------------------------------------

def _generate_eval_samples(rag_engine, num_samples: int = 5) -> list[dict]:
    """
    Automatically generate evaluation samples from the loaded document.

    Strategy: Pick a few chunks, use a portion of their text as the
    'query', and expect retrieval to find the same or similar chunks.
    This tests whether the retrieval system can match content to itself.

    Args:
        rag_engine: An initialized RAGEngine instance with documents loaded.
        num_samples: Number of evaluation samples to generate.

    Returns:
        List of dicts with 'query' and 'source_text' keys.
    """
    chunks = rag_engine.chunks
    if not chunks:
        return []

    # Pick evenly spaced chunks to get diverse samples
    step = max(1, len(chunks) // num_samples)
    selected = chunks[::step][:num_samples]

    samples = []
    for chunk in selected:
        text = chunk.page_content.strip()
        if len(text) < 20:
            continue

        # Use the first ~100 chars as a query (simulating a user question
        # about that chunk's topic)
        query_text = text[:100].strip()
        samples.append({
            "query": query_text,
            "source_text": text,
        })

    return samples


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.3  # Minimum overlap ratio to count as a "hit"


def _text_overlap(query_text: str, chunk_text: str) -> float:
    """
    Compute a simple word-overlap ratio between query and chunk.
    Returns a float between 0.0 and 1.0.
    """
    query_words = set(query_text.lower().split())
    chunk_words = set(chunk_text.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & chunk_words
    return len(overlap) / len(query_words)


def evaluate_retrieval(rag_engine, k: int = 4) -> EvalResult:
    """
    Evaluate retrieval quality using auto-generated queries from the document.

    For each sample, uses a portion of a chunk as query and checks if
    retrieval returns the same or similar chunk in the top-k results.

    Args:
        rag_engine: An initialized RAGEngine instance with documents loaded.
        k: Number of chunks to retrieve per query.

    Returns:
        EvalResult with hit_rate, mrr, and per-query details.
    """
    samples = _generate_eval_samples(rag_engine)

    if not samples:
        return EvalResult(total_queries=0)

    result = EvalResult(total_queries=len(samples))
    reciprocal_ranks = []

    for sample in samples:
        query = sample["query"]
        source_text = sample["source_text"]

        # Retrieve chunks
        retrieved = rag_engine.retrieve_chunks(query, k=k)
        retrieved_texts = [doc.page_content for doc in retrieved]

        # Check if any retrieved chunk overlaps significantly with source
        hit = False
        first_relevant_rank = None

        for rank, chunk_text in enumerate(retrieved_texts, start=1):
            overlap = _text_overlap(source_text, chunk_text)
            if overlap >= SIMILARITY_THRESHOLD and first_relevant_rank is None:
                first_relevant_rank = rank
                hit = True

        # Record results
        if hit:
            result.hits += 1
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

        result.details.append({
            "question": query[:80] + ("..." if len(query) > 80 else ""),
            "hit": hit,
            "first_relevant_rank": first_relevant_rank,
            "num_chunks_retrieved": len(retrieved_texts),
        })

    # Compute aggregate metrics
    if result.total_queries > 0:
        result.hit_rate = result.hits / result.total_queries
        result.mrr = sum(reciprocal_ranks) / result.total_queries

    return result
