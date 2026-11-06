from src.models import (
    ChunkHolder,
)

import bm25s

from src.corpus import corpus_manager, sources_manager


def content_manager(
    sources: list[ChunkHolder], max_chunk_size: int
) -> list[str]:
    """Extract chunk text after checking character limits.

    Args:
        sources: Source-aware chunks.
        max_chunk_size: Maximum permitted characters per chunk.

    Returns:
        Chunk texts in their original indexing order.

    Raises:
        ValueError: If a chunk is empty or exceeds the size limit.
    """
    chunks = []
    for s in sources:
        if len(s.text) <= 0:
            raise ValueError("chunk must not be empty")
        elif not (len(s.text) <= max_chunk_size):
            raise ValueError("chunk to large")
        chunks.append(s.text)
    return chunks


def bm25_tester(
    sources: list[ChunkHolder], max_chunk_size: int
) -> None:
    """Build and save a BM25 index with chunk metadata.

    Args:
        sources: Chunk holders to index and persist.
        max_chunk_size: Maximum allowed length of each chunk.
    """
    chunks = content_manager(sources, max_chunk_size)
    tokens = bm25s.tokenize(chunks)
    meta_data = [chunk.model_dump() for chunk in sources]
    retriver = bm25s.BM25()
    retriver.index(tokens)
    retriver.save("data/processed/index", corpus=meta_data)


def index(max_chunk_size: int = 2000) -> None:
    """Index Markdown and Python sources in the raw corpus.

    Args:
        max_chunk_size: Upper limit for characters in each chunk.

    Raises:
        ValueError: If the requested chunk size is outside 1-2000.
    """
    if not (1 <= max_chunk_size <= 2000):
        raise ValueError("max_size must be between 1 and 2000")
    data = corpus_manager()
    sources = sources_manager(data, max_chunk_size)
    bm25_tester(sources, max_chunk_size)
