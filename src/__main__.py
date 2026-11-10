import fire
import sys

from src.evaluator import evaluate
from src.indexer import index
from src.retriever import search, search_dataset
from src.generator import answer, answer_dataset
from src.semantic import semantic_search


if __name__ == "__main__":
    try:
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
        semantic_search("How does vLLM manage its KV cache?", 5)
    except (ValueError, OSError, Exception) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from None
