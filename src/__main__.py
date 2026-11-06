import fire

from src.evaluator import evaluate
from src.indexer import index
from src.retriever import search, search_dataset
from src.generator import answer, answer_dataset


if __name__ == "__main__":
    fire.Fire(
        {
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate,
        }
    )
