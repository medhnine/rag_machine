from pathlib import Path

from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)


from src.models import (
    ChunkHolder
)


def file_manager(p: str | Path) -> dict[str, list[Path]]:
    """Find Markdown and Python files recursively under a directory.

    Args:
        p: Root directory to scan.

    Returns:
        Paths grouped by the "md_data" and "py_data" keys.
    """
    res = {}
    path_obj = Path(p)
    md_files = list(path_obj.rglob("*.md"))
    py_files = list(path_obj.rglob("*.py"))
    res["md_data"] = md_files
    res["py_data"] = py_files
    return res


def read_file(path: Path) -> str:
    """Read a UTF-8 source file.

    Args:
        path: Path of the file to read.

    Returns:
        The full file content.

    Raises:
        ValueError: If the file cannot be read.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"an error aquired while opening the file: {e}")
    return text


def corpus_manager() -> dict[str, list[tuple[Path, str]]]:
    """Load the raw corpus as file-path and text pairs.

    Returns:
        Markdown and Python records grouped by file type.
    """
    data = file_manager("data/raw")
    files = {}
    for key, value in data.items():
        result = []
        for file in value:
            text = read_file(file)
            if text is not None:
                result.append((file, text))
        files[key] = result
    return files


def md_text_spliter(
    data: tuple[Path, str], size: int
) -> list[ChunkHolder]:
    """Split one Markdown file into offset-aware chunk holders.

    Args:
        data: Original file path and complete source text.
        size: Requested maximum chunk size.

    Returns:
        Chunks retaining file paths, offsets, and original text.

    Raises:
        ValueError: If a chunk does not match its source slice.
    """
    list_chunkholder = []
    overlap = int(size / 4)
    chunk_result = MarkdownTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
        keep_separator=True,
    )
    document = chunk_result.create_documents([data[1]])
    for doc in document:
        file_path = str(data[0])
        text = doc.page_content
        start_index = doc.metadata["start_index"]
        end_index = start_index + len(text)
        if text != data[1][start_index:end_index]:
            raise ValueError("validtion error")
        chunk_obj = ChunkHolder(
            file_path=file_path,
            first_character_index=start_index,
            last_character_index=end_index,
            text=text,
        )
        list_chunkholder.append(chunk_obj)
    return list_chunkholder


def py_code_spliter(
    data: tuple[Path, str], size: int
) -> list[ChunkHolder]:
    """Split one Python file into offset-aware chunk holders.

    Args:
        data: Original file path and complete source text.
        size: Requested maximum chunk size.

    Returns:
        Chunks retaining file paths, offsets, and original text.

    Raises:
        ValueError: If a chunk does not match its source slice.
    """
    list_chunkholder = []
    overlap = int(size / 4)
    chunk_result = PythonCodeTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
        keep_separator=True,
    )
    document = chunk_result.create_documents([data[1]])
    for doc in document:
        file_path = str(data[0])
        text = doc.page_content
        start_index = doc.metadata["start_index"]
        end_index = start_index + len(text)
        if text != data[1][start_index:end_index]:
            raise ValueError("validtion error")
        chunk_obj = ChunkHolder(
            file_path=file_path,
            first_character_index=start_index,
            last_character_index=end_index,
            text=text,
        )
        list_chunkholder.append(chunk_obj)
    return list_chunkholder


def sources_manager(
    data: dict[str, list[tuple[Path, str]]], max_size: int
) -> list[ChunkHolder]:
    """Build chunk holders for each corpus file type.

    Args:
        data: Corpus records grouped by Markdown/Python file type.
        max_size: Requested maximum chunk size.

    Returns:
        A flattened list of source-aware chunks.
    """
    sources = []
    for key, value in data.items():
        for tp in value:
            if key == "md_data":
                res = md_text_spliter(tp, max_size)
                sources.extend(res)
            elif key == "py_data":
                sources.extend(py_code_spliter(tp, max_size))
    return sources
