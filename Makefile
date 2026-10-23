MAX_CHUNK_SIZE ?=2000
QUERY ?= vllm uses
DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_code_public.json
K ?= 5
SAVE_DIRECTORY ?= data/datasets/AnsweredQuestions


index:
	uv run python3 -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python3 -m src search --query "$(QUERY)" --k $(K)

search_dataset:
	uv run python3 -m src search_dataset --dataset_path $(DATASET_PATH) --k $(K) --save_directory $(SAVE_DIRECTORY) 

install:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
