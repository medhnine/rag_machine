from pathlib import Path
from typing import Sequence

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults
)

from src.utils import read_json_file


def compare_sources(
    retrived: MinimalSource, correct: MinimalSource
) -> bool:
    """Determine whether two source spans count as a match.

    Two spans match when their file paths agree and the
    intersection-over-union of their character ranges is at least 0.05.

    Args:
        retrived: BM25-retrieved source location.
        correct: Ground-truth source location.

    Returns:
        True when the ranges satisfy the matching rule.
    """
    if retrived.file_path != correct.file_path:
        return False
    r_start = retrived.first_character_index
    r_end = retrived.last_character_index
    c_start = correct.first_character_index
    c_end = correct.last_character_index

    if r_start >= r_end or c_start >= c_end:
        return False

    area_0f_intersection = min(c_end, r_end) - max(c_start, r_start)
    if area_0f_intersection <= 0:
        return False

    area_correct = c_end - c_start
    area_retrived = r_end - r_start

    area_of_union = area_correct + area_retrived - area_0f_intersection

    intersection_over_union = area_0f_intersection / area_of_union

    if intersection_over_union < 0.05:
        return False

    return True


def calculate_recall(
    correct_sources: Sequence[MinimalSource],
    retrieved_sources: Sequence[MinimalSource],
    k: int,
) -> float:
    """Calculate top-k recall for one question.

    Each correct source counts once if any top-k source matches it.

    Args:
        correct_sources: Ground-truth source locations.
        retrieved_sources: Ranked BM25 source locations.
        k: Number of retrieved locations to consider.

    Returns:
        Proportion of ground-truth sources recovered.

    Raises:
        ValueError: If no correct sources are provided.
    """
    found = 0
    if len(correct_sources) == 0:
        raise ValueError("empty correct_sources")
    for source in correct_sources:
        for retrieved in retrieved_sources[:k]:
            if compare_sources(retrieved, source):
                found += 1
                break
    print(
        f"recall: {found}/{len(correct_sources)} = "
        f"{(found / len(correct_sources)) * 100}"
    )
    return found / len(correct_sources)


def evaluate(
    student_search_results_path: str | Path, dataset_path: str | Path
) -> None:
    """Compute and print mean Recall@k for saved search results.

    Matches question IDs to their ground-truth source locations,
    then averages the recall values over student questions.

    Args:
        student_search_results_path: Saved retrieval results JSON.
        dataset_path: AnsweredQuestions ground-truth JSON.

    Raises:
        ValueError: If ground truth is missing or questions are unknown.
    """
    dataset = RagDataset.model_validate(read_json_file(dataset_path))

    student_results = StudentSearchResults.model_validate(
        read_json_file(student_search_results_path)
    )
    total_questions = len(student_results.search_results)
    if total_questions == 0:
        raise ValueError("No questions to evaluate")
    k = student_results.k
    extracted_data = {}
    result = 0.0
    for item in dataset.rag_questions:
        if not isinstance(item, AnsweredQuestion):
            raise ValueError("error in evalute")
        extracted_data[item.question_id] = item.sources

    for student_item in student_results.search_results:
        if not isinstance(student_item, MinimalSearchResults):
            raise ValueError("error in evalute")
        question_id = student_item.question_id
        sources = extracted_data.get(question_id)
        if sources is None:
            raise ValueError(f"Unknown question id {question_id}")
        retrieved_sources = student_item.retrieved_sources
        result += calculate_recall(sources, retrieved_sources, k)
    print(f"recall@{k}: {((result / total_questions) * 100):1f}%")
