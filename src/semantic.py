from sentence_transformers import SentenceTransformer
from src.retriever import results_manger
from pathlib import Path

import numpy as np
import bm25s

class SemanticRetriever:
    def __init__(self):
        self.retriever = bm25s.BM25.load(
            "data/processed/index",
            load_corpus=True
        )

        self.corpus = self.retriever.corpus

        self.embeddings = np.load("data/processed/semantic/embeddings.npy")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        if self.embeddings.shape[0] != len(self.corpus):
            raise ValueError("incomplete embedding index")

    def semantic_search(self, query, k):
        if not isinstance(query, str) or not len(query.strip()):
            raise ValueError("error query must be str and not empty")
        elif not isinstance(k, int) or k <= 0:
            raise ValueError("error k must be int and > 0")
        query_ids = semantic_encode(query, self.model)
        scores = self.embeddings @ query_ids
        k_indices = np.argsort(scores)[::-1][:k]
        top_chunks = []
        for index in k_indices:
            top_chunks.append(self.corpus[int(index)])
        result = results_manger(top_chunks)
        print(result[0].file_path)
        print(result[0].first_character_index)
        print(result[0].last_character_index)
        print(len(result))
        return result


def semantic_encode(data, model):
    embeddings =  model.encode(
        data,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings

def semantic_embedding():
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    texts= []
    for chunk in retriver.corpus:
        texts.append(chunk["text"])
    model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
    )
    embeddings = semantic_encode(texts, model)
    output_dir = Path("data/processed/semantic")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    embedded_result = np.load("data/processed/semantic/embeddings.npy")
    if embedded_result.shape[0] != len(texts):
        raise ValueError("incomplete embedding index")
    print(embedded_result.shape)
    print(embedded_result.dtype)
    print(f"{embedded_result.shape[0]} == {len(texts)}")
    print(embedded_result.shape[1])
    assert embedded_result.shape[0] == len(texts)
    assert embedded_result.shape[1] == 384

def semantic_search(query, k):
    embeddings = np.load("data/processed/semantic/embeddings.npy")
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    corpus = retriver.corpus
    if not isinstance(query, str) and not len(query.strip()):
        raise ValueError("error query must be str and not empty")
    elif not isinstance(k, int) or k <= 0:
        raise ValueError("error k must be int and > 0")
    if len(corpus) != len(embeddings):
        raise ValueError("error len(corpus) != len(embeddings)")
    model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
    )
    query_ids = semantic_encode(query, model)
    scores = embeddings @ query_ids
    k_indices = np.argsort(scores)[::-1][:k]
    top_chunks = []
    for index in k_indices:
        top_chunks.append(corpus[int(index)])
    result = results_manger(top_chunks)
    print(result[0].file_path)
    print(result[0].first_character_index)
    print(result[0].last_character_index)
    print(len(result))
    return result
    


# semantic_embedding()
