from pathlib import Path
from typing import cast
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter, PythonCodeTextSplitter
import fire
from src.utils import ChunkHolder

class Corpus:
    def task(task, time):
        print(f"thats your task: {task}, time to finish is: {time}")


    def file_manager(p):
        res = {}
        path_obj = Path(p)
        md_files = list(path_obj.rglob("*.md"))
        py_files = list(path_obj.rglob("*.py"))
        # res["md_files"]=  len(md_files)
        # res["py_files"] = len(py_files)
        res["md_data"] = md_files
        res["py_data"] = py_files
        return res


    def read_file(path):
        try:
            text = path.read_text(encoding="utf-8")
            # print(type(text))
            # print(len(text))
            # print(text[:500])
        except OSError as e:
            raise ValueError(f"an error aquired while opening the file: {e}")
        return text


    def corpus_mangaer():
        data = file_manager("data/raw/vllm-0.10.1")
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
        chunk_result = MarkdownTextSplitter(chunk_size = size, chunk_overlap=50, length_function=len, add_start_index=True, keep_separator=True)
        document = chunk_result.create_documents([data[1]])
        for doc in document:
            chunk_obj = ChunkHolder
            chunk_obj.file_path = data[0]
            text = doc.page_content
            start_index = doc.metadata["start_index"]
            end_index = start_index + len(text)
            if text != data[1][start_index:end_index]:
                raise ValueError("validtion error")
            chunk_obj.first_character_index = start_index
            chunk_obj.last_character_index = end_index
            chunk_obj.text = text
            list_chunkholder.append(chunk_obj)
        return list_chunkholder
        
    def py_code_spliter(data, size):
        list_chunkholder = []
        chunk_result = PythonCodeTextSplitter(chunk_size = size, chunk_overlap=50, length_function=len, add_start_index=True, keep_separator=True)
        document = chunk_result.create_documents([data[1]])
        for doc in document:
            chunk_obj = ChunkHolder
            chunk_obj.file_path = data[0]
            text = doc.page_content
            start_index = doc.metadata["start_index"]
            end_index = start_index + len(text)
            if text != data[1][start_index:end_index]:
                raise ValueError("validtion error")
            chunk_obj.first_character_index = start_index
            chunk_obj.last_character_index = end_index
            chunk_obj.text = text
            list_chunkholder.append(chunk_obj)
        return list_chunkholder


# def chunk_manger(data):
#     for key, value in data.items():
#         pass


if __name__ == "__main__":
    data = corpus_mangaer()
    sources = []
    spliter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap=0)
    for key, value in data.items():
        for tp in value:
            if key == "md_data":
                sources.append(md_text_spliter(tp, 2000))
            elif key == "py_data":
                sources.append(py_code_spliter(tp, 2000))
    print(sources[0])
    print(sources[0][0].text[:300]) 
    print(sources[0][0].first_character_index)
    print(sources[0][0].last_character_index)
    print(sources[0][0].file_path)

    
    # text = spliter.split_text(data["md_data"][1][1])
    # print(text[0])
    # fire.Fire({"task": task,})
    # res = file_manager("data/raw/vllm-0.10.1")
    # read_file(res["md_data"][1])
    # print(res.read_text(encoding="utf-8"))