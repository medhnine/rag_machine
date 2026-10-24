from pathlib import Path
from typing import cast
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter
import fire
import bm25s
from src.utils import ChunkHolder, MinimalSource, MinimalSearchResults, UnansweredQuestion, StudentSearchResults
import json
import pprint
import os

def file_manager(p):
    res = {}
    path_obj = Path(p)
    md_files = list(path_obj.rglob("*.md"))
    py_files = list(path_obj.rglob("*.py"))
    res["md_data"] = md_files
    res["py_data"] = py_files
    return res


def read_file(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"an error aquired while opening the file: {e}")
    return text


def corpus_manager():
    data = file_manager("data/raw")
    files = {}
    for key, value, in data.items():
        result = []
        for file in value:
            text = read_file(file)
            if text is not None:
                result.append((file, text))
        files[key] = result
    return files


def md_text_spliter(data, size):
    list_chunkholder = []
    chunk_result = MarkdownTextSplitter(chunk_size = size, chunk_overlap=500, length_function=len, add_start_index=True, keep_separator=True)
    document = chunk_result.create_documents([data[1]])
    for doc in document:
        file_path = str(data[0])
        text = doc.page_content
        start_index = doc.metadata["start_index"]
        end_index = start_index + len(text)
        if text != data[1][start_index:end_index]:
            raise ValueError("validtion error")
        chunk_obj = ChunkHolder(file_path=file_path, first_character_index=start_index, last_character_index=end_index, text=text)
        list_chunkholder.append(chunk_obj)
    return list_chunkholder


def py_code_spliter(data, size):
    list_chunkholder = []
    chunk_result = PythonCodeTextSplitter(chunk_size = size, chunk_overlap=500, length_function=len, add_start_index=True, keep_separator=True)
    document = chunk_result.create_documents([data[1]])
    for doc in document:
        file_path = str(data[0])
        text = doc.page_content
        start_index = doc.metadata["start_index"]
        end_index = start_index + len(text)
        if text != data[1][start_index:end_index]:
            raise ValueError("validtion error")
        chunk_obj = ChunkHolder(file_path=file_path, first_character_index=start_index, last_character_index=end_index, text=text)
        list_chunkholder.append(chunk_obj)
    return list_chunkholder


def bm25_tester(sources, max_chunk_size):
    chunks = content_manager(sources, max_chunk_size)
    tokens = bm25s.tokenize(chunks)
    meta_data = [chunk.model_dump() for chunk in sources]
    # query = "vLLM is a fast ?"
    # query_token = bm25s.tokenize(query)
    retriver = bm25s.BM25()
    retriver.index(tokens)
    # result = retriver.retrieve(query_token, k=2)
    retriver.save("data/processed/index", corpus=meta_data)
    # return result


def sources_manager(data, max_size):
    sources = []
    for key, value in data.items():
        for tp in value:
            if key == "md_data":
                res = md_text_spliter(tp, max_size)
                sources.extend(res)
            elif key == "py_data":
                sources.extend(py_code_spliter(tp, max_size))
    return sources


def content_manager(sources, max_chunk_size):
    chunks = []
    for s in sources:
        if len(s.text) <= 0:
            raise ValueError("chunk must not be empty")
        elif not (len(s.text) <= max_chunk_size):
            raise ValueError("chunk to large")
        chunks.append(s.text)
    return chunks


def results_manger(data):
    results = []
    for res in data:
        file = res["file_path"]
        first = res["first_character_index"]
        last = res["last_character_index"]
        results.append(MinimalSource(file_path=file, first_character_index=first, last_character_index=last))
    return results


def read_json_file(path):
    try:
        with open(path, "r") as f:
            result = json.load(f)
            return result
    except OSError as e:
        raise ValueError(f"an error {e}")


def search(query, k=5):
    if len(query) == 0:
        raise ValueError("query must not be empty")
    elif k <= 0:
        raise ValueError("k must not be <= 0")
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    corpus = retriver.corpus
    query_tokens = bm25s.tokenize(query)
    results, scores = retriver.retrieve(query_tokens, k=k)
    sources = results_manger(results[0])
    id = UnansweredQuestion(question=query).question_id
    return MinimalSearchResults(question_id=id, question=query, retrieved_sources=sources).model_dump_json(indent=2)


def query_search(query, k, id, retriver):
    tokens = bm25s.tokenize(query)
    results, scores = retriver.retrieve(tokens, k=k)
    sources = results_manger(results[0])
    return MinimalSearchResults(question_id=id, question=query, retrieved_sources=sources)

def json_save(dataset_path, data, save_directory):
    output_dir = Path(save_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(dataset_path).name
    output_path = output_dir / file_name
    output_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    path = Path(output_path)
    valid = data.model_validate_json(path.read_text(encoding="utf-8"))


def search_dataset(dataset_path, k, save_directory):
    query_data = read_json_file(dataset_path)
    data = query_data["rag_questions"]
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    MinimalSearchResults_data = []
    for query in data:
        question_id = query["question_id"]
        question = query["question"]
        sources = query_search(question, k, question_id, retriver)
        MinimalSearchResults_data.append(sources)
    data = StudentSearchResults(search_results=MinimalSearchResults_data, k=k)
    json_save(dataset_path ,data, save_directory)
# /goinfre/mohhnine/rag_git/data/output/search_results/UnansweredQuestions/dataset_docs_public.json

def index(max_chunk_size=2000):
    if not (1 <= max_chunk_size <= 2000):
        raise ValueError("max_size must be between 1 and 2000")
    data = corpus_manager()
    sources = sources_manager(data, max_chunk_size)
    bm25_result = bm25_tester(sources, max_chunk_size)
    # result, scores = bm25_result


if __name__ == "__main__":
    max_size = fire.Fire({"index": index, "search": search, "search_dataset": search_dataset})
    # data = corpus_manager()
    # sources = sources_manager(data)
    # chunks = content_manager(sources)
    # bm25_result = bm25_tester(chunks)
    # print(bm25_result)
    # result, scores = bm25_result
    # print(chunks[result[0][0]])
    # text = spliter.split_text(data["md_data"][1][1])
    # print(text[0])
    # fire.Fire({"task": task,})
    # res = file_manager("data/raw/vllm-0.10.1")
    # read_file(res["md_data"][1])
    # print(res.read_text(encoding="utf-8"))


