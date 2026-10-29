from pathlib import Path
from typing import cast
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter
import fire
import bm25s
from src.utils import ChunkHolder, MinimalSource, MinimalSearchResults, UnansweredQuestion, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer
import json
import pprint
import os
from transformers import pipeline
import torch
from uuid import uuid4


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
        with open(path, "r", encoding="utf-8") as f:
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


def context_maker(inputs, k, query):
    if len(inputs) == 0:
        raise ValueError("inputs data are not enough for context generation")
    context = "You are an assistant answering questions.\n\n\nUse the provided context to answer the question.\nIf the context does not contain enough information, say so.\nDo not invent information.\n\n"
    context += "CONTEXT:\n"
    for i in range(0, k):
        context += f"\n\n[source {i+1}]\n"
        context += inputs[i]["text"]
    context += "\n\n\nQUESTION:\n"
    context += query
    context += "\n\nANSWER:"
    return context


def llm_tester(context, llm):
    if llm is None:
        llm = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device_map="auto")
    # model = llm.model
    # model_tokenizer = llm.tokenizer

    messages = [
        {
            "role": "user",
            "content": context + "\n/no_think"
        } 
    ]
    result = llm(
        messages,
        max_new_tokens=256,
        return_full_text=False
    )
    return result[0]["generated_text"].strip("<think>\n</think>\n")
    # inputs = model_tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_promt=True,
    #     enable_thinking=False,
    #     return_tensors="pt",
    #     return_dict=True
    # ).to(model.device)
    # pprint.pp(inputs)
    # exit(0)
    # ids = model_tokenizer.encode("the capital of spain is madrid and the capital of france is par")
    # inputs = torch.tensor([ids], device=model.device)
    # result = model.generate(input_ids=inputs)
    # print(model_tokenizer.decode(result))
    # res = model.generate()
    # print(model_tokenizer.decode(res))


def answer(query, k):
    if len(query.strip()) == 0:
        raise ValueError("query must not be empty")
    elif k <= 0:
        raise ValueError("k must not be <= 0")
    retriver = bm25s.BM25.load("data/processed/index", load_corpus=True)
    tokens = bm25s.tokenize(query)
    results, scores = retriver.retrieve(tokens, k=k)
    context = context_maker(results[0], k, query)
    result = llm_tester(context, None)
    answer_result = MinimalAnswer(
    question_id=str(uuid4()),
    question=query,
    retrieved_sources=results_manger(results[0]),
    answer=result
    )
    return answer_result

def build_context(text, query):
    if len(text) == 0:
        raise ValueError("inputs data are not enough for context generation")
    context = "You are an assistant answering questions.\n\n\nUse the provided context to answer the question.\nIf the context does not contain enough information, say so.\nDo not invent information.\n\n"
    context += "CONTEXT:\n"
    for i in range(0, len(text)):
        context += f"\n\n[source {i+1}]\n"
        context += text[i]
    context += "\n\n\nQUESTION:\n"
    context += query
    context += "\n\nANSWER:"
    return context


def minimal_source_reader(source):
    start = source.first_character_index
    end = source.last_character_index
    with open(source.file_path, "r") as f:
        length = end - start
        text = f.read()
    if start < 0 or end <= start or end > len(text):
        raise ValueError(f"Invalid source offsets: {source.file_path}")
    return text[start:end]

def minimal_search_results_extractor(search_results):
    list_minimal_answer = []
    llm = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device_map="auto")
    for item in search_results:
        question = item.question
        print(question)
        print()
        question_id = item.question_id
        retrieved_sources = item.retrieved_sources
        texts = []
        for source in retrieved_sources:
            text = minimal_source_reader(source)
            texts.append(text)
        context = build_context(texts, question)
        answer = llm_tester(context, llm)
        print(answer)
        list_minimal_answer.append(MinimalAnswer(
            question_id=question_id, 
            question=question, 
            retrieved_sources=retrieved_sources,
            answer=answer
        ))
    return list_minimal_answer
        

def answer_dataset(student_search_results_path, save_directory):
    data = StudentSearchResults.model_validate(read_json_file(student_search_results_path))
    search_results = data.search_results
    k = data.k
    search_results = minimal_search_results_extractor(search_results)
    final_result = StudentSearchResultsAndAnswer(search_results=search_results, k=k)
    json_save(student_search_results_path, final_result, save_directory)
    return final_result


def index(max_chunk_size=2000):
    if not (1 <= max_chunk_size <= 2000):
        raise ValueError("max_size must be between 1 and 2000")
    data = corpus_manager()
    sources = sources_manager(data, max_chunk_size)
    bm25_result = bm25_tester(sources, max_chunk_size)
    # result, scores = bm25_result


if __name__ == "__main__":
    max_size = fire.Fire({"index": index, "search": search, "search_dataset": search_dataset, "answer": answer, "answer_dataset": answer_dataset})
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
