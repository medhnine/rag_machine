from typing import Iterable, Any, Mapping
from pathlib import Path

import bm25s

from src.utils import read_json_file, json_save
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
)


def results_manger(
    data: Iterable[Mapping[str, Any]],
) -> list[MinimalSource]:
    """Convert retrieved BM25 records into minimal sources.

    Args:
        data: Retrieved chunk metadata mappings.

    Returns:
        File paths and character offsets without chunk text.
    """
    results = []
    for res in data:
        file = res["file_path"]
        first = res["first_character_index"]
        last = res["last_character_index"]
        results.append(
            MinimalSource(
                file_path=file,
                first_character_index=first,
                last_character_index=last,
            )
        )
    return results


def query_search(
    query: str, k: int, id: str, retriver: Any
) -> MinimalSearchResults:
    """Run one BM25 query using an already loaded retriever.

    Args:
        query: Question text to retrieve against.
        k: Number of requested sources.
        id: Existing question identifier to preserve.
        retriver: Loaded bm25s BM25 object, including its corpus.

    Returns:
        Retrieved minimal source locations for this question.
    """
    tokens = bm25s.tokenize(query)
    results, scores = retriver.retrieve(tokens, k=k)
    sources = results_manger(results[0])
    return MinimalSearchResults(
        question_id=id,
        question=query,
        retrieved_sources=sources,
    )


def search(query: str, k: int = 5) -> str:
    """Retrieve top-k sources for a single CLI query.

    Args:
        query: Text to search for in the indexed corpus.
        k: Maximum number of retrieved sources.

    Returns:
        A JSON-serialized MinimalSearchResults record.

    Raises:
        ValueError: If the query is empty or k is nonpositive.
    """
    if len(query) == 0:
        raise ValueError("query must not be empty")
    elif k <= 0:
        raise ValueError("k must not be <= 0")
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    query_tokens = bm25s.tokenize(query)
    results, scores = retriver.retrieve(query_tokens, k=k)
    sources = results_manger(results[0])
    id = UnansweredQuestion(question=query).question_id
    return MinimalSearchResults(
        question_id=id,
        question=query,
        retrieved_sources=sources,
    ).model_dump_json(indent=2)


def search_dataset(
    dataset_path: str | Path, k: int, save_directory: str | Path
) -> None:
    """Retrieve sources for every question in a dataset.

    Args:
        dataset_path: JSON file containing rag_questions.
        k: Number of retrieved sources per question.
        save_directory: Output folder for StudentSearchResults.
    """
    query_data = read_json_file(dataset_path)
    data = query_data["rag_questions"]
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    MinimalSearchResults_data = []
    for query in data:
        question_id = query["question_id"]
        question = query["question"]
        sources = query_search(question, k, question_id, retriver)
        MinimalSearchResults_data.append(sources)
    data = StudentSearchResults(
        search_results=MinimalSearchResults_data, k=k
    )
    json_save(dataset_path, data, save_directory)
