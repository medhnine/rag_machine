import json
from pathlib import Path
from typing import Any


from pydantic import BaseModel


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Args:
        path: Path to the JSON input.

    Returns:
        A decoded dictionary.

    Raises:
        ValueError: If opening the file fails.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
            return result
    except OSError as e:
        raise ValueError(f"an error {e}")


def json_save(
    dataset_path: str | Path,
    data: BaseModel,
    save_directory: str | Path,
) -> None:
    """Serialize a Pydantic result to a named output JSON file.

    Args:
        dataset_path: Source filename used to name the output file.
        data: Pydantic result to serialize and validate.
        save_directory: Folder in which to write the JSON file.
    """
    output_dir = Path(save_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(dataset_path).name
    output_path = output_dir / file_name
    output_path.write_text(
        data.model_dump_json(indent=2), encoding="utf-8"
    )
    path = Path(output_path)
    data.model_validate_json(path.read_text(encoding="utf-8"))
