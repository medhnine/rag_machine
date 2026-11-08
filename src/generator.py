from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from typing import Any, Mapping, Sequence

import bm25s
from transformers import pipeline

from src.utils import read_json_file, json_save
from src.retriever import results_manger
from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSearchResults,
)


def context_maker(
    inputs: Sequence[Mapping[str, Any]], k: int, query: str
) -> str:
    """Build an augmented prompt from ranked BM25 chunk records.

    Args:
        inputs: Retrieved chunks containing a "text" field.
        k: Number of chunks to include in the prompt.
        query: User question appended after the context.

    Returns:
        Instructions, ranked context, and question as one string.
    """
    if not isinstance(query, str) or len(query.strip()) == 0:
        raise ValueError("query must be valid str and not be empty")
    elif not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must not be <= 0")
    if len(inputs) == 0:
        raise ValueError("inputs data are not enough for context generation")
    context = (
        "You are an assistant answering questions.\n\n\n"
        "Use the provided context to answer the question.\n"
        "If the context does not contain enough information, say so.\n"
        "Do not invent information.\n\n"
    )
    context += "CONTEXT:\n"
    for i in range(0, k):
        context += f"\n\n[source {i+1}]\n"
        context += inputs[i]["text"]
    context += "\n\n\nQUESTION:\n"
    context += query
    context += "\n\nANSWER:"
    return context


def build_context(text: Sequence[str], query: str) -> str:
    """Build a question prompt from reconstructed source strings.

    Args:
        text: Chunk texts read from their original files.
        query: Question to answer using those chunks.

    Returns:
        Instruction and context prompt for generation.
    """
    if not isinstance(query, str) or len(query.strip()) == 0:
        raise ValueError("query must be valid str and not be empty")
    if len(text.strip()) == 0:
        raise ValueError("inputs data are not enough for context generation")
    context = (
        "You are an assistant answering questions.\n\n\n"
        "Use the provided context to answer the question.\n"
        "If the context does not contain enough information, say so.\n"
        "Do not invent information.\n\n"
    )
    context += "CONTEXT:\n"
    for i in range(0, len(text)):
        context += f"\n\n[source {i+1}]\n"
        context += text[i]
    context += "\n\n\nQUESTION:\n"
    context += query
    context += "\n\nANSWER:"
    return context


def llm_tester(context: str, llm: Any | None) -> str:
    """Generate an answer from a context prompt using Qwen.

    Args:
        context: Augmented prompt to send to the model.
        llm: Existing text-generation pipeline, or None to load one.

    Returns:
        Generated text with surrounding whitespace removed.
    """
    if llm is None:
        llm = pipeline(
            "text-generation",
            model="Qwen/Qwen3-0.6B",
            device_map="auto",
        )

    messages = [
        {
            "role": "user",
            "content": context + "\n/no_think",
        }
    ]
    result = llm(
        messages,
        max_new_tokens=256,
        return_full_text=False,
    )
    final_answer = str(result[0]["generated_text"]).strip()
    return final_answer


def minimal_source_reader(source: MinimalSource) -> str:
    """Read the original character span specified by a source.

    Args:
        source: File path and start/end character offsets.

    Returns:
        Text within the stored source span.

    Raises:
        ValueError: If the offsets are outside the file.
    """
    start = source.first_character_index
    end = source.last_character_index
    try:
        with open(source.file_path, "r") as f:
            text = f.read()
    except OSError as e:
        raise ValueError(f"an error aquired while opening the file: {e}")
    if start < 0 or end <= start or end > len(text):
        raise ValueError(f"Invalid source offsets: {source.file_path}")
    return text[start:end]


def minimal_search_results_extractor(
    search_results: Sequence[MinimalSearchResults],
) -> list[MinimalAnswer]:
    """Generate MinimalAnswer objects from saved search results.

    Loads the Qwen pipeline once and reuses it across questions.

    Args:
        search_results: Validated retrieval results for a dataset.

    Returns:
        One generated answer and its sources per input question.
    """
    list_minimal_answer = []
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen3-0.6B",
        device_map="auto",
    )
    for item in tqdm(search_results, desc="Generating answers"):
        question = item.question
        question_id = item.question_id
        retrieved_sources = item.retrieved_sources
        texts = []
        for source in retrieved_sources:
            text = minimal_source_reader(source)
            texts.append(text)
        context = build_context(texts, question)
        answer = llm_tester(context, llm)
        list_minimal_answer.append(
            MinimalAnswer(
                question_id=question_id,
                question=question,
                retrieved_sources=retrieved_sources,
                answer=answer,
            )
        )
    return list_minimal_answer


def answer(query: str, k: int) -> str:
    """Retrieve relevant chunks and answer one user question.

    Args:
        query: Input question to answer.
        k: Number of BM25 context chunks.

    Returns:
        JSON-serialized MinimalAnswer with source metadata.
    """
    if not isinstance(query, str) or len(query.strip()) == 0:
        raise ValueError("query must be str and not empty")
    elif not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must not be <= 0")
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    tokens = bm25s.tokenize(query)
    results, scores = retriver.retrieve(tokens, k=k)
    context = context_maker(results[0], k, query)
    result = llm_tester(context, None)
    if result.startswith("<think>\n\n</think>\n\n"):
        result = result.split("<think>\n\n</think>\n\n", 1)[1].strip()
    answer_result = MinimalAnswer(
        question_id=str(uuid4()),
        question=query,
        retrieved_sources=results_manger(results[0]),
        answer=result,
    )
    return answer_result.model_dump_json(indent=2)


def answer_dataset(
    student_search_results_path: str | Path, save_directory: str | Path
) -> StudentSearchResultsAndAnswer:
    """Generate and persist answers for saved retrieval results.

    Args:
        student_search_results_path: Input StudentSearchResults JSON.
        save_directory: Directory in which to save answer JSON.

    Returns:
        A validated StudentSearchResultsAndAnswer object.
    """
    data = StudentSearchResults.model_validate(
        read_json_file(student_search_results_path)
    )
    search_results = data.search_results
    k = data.k
    answer_results = minimal_search_results_extractor(search_results)
    final_result = StudentSearchResultsAndAnswer(
        search_results=answer_results, k=k
    )
    json_save(student_search_results_path, final_result, save_directory)
    return final_result
