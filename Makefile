MAX_CHUNK_SIZE ?=2000
QUERY ?= what is my name
K ?= 5

index:
	uv run python3 -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python3 -m src search --query "$(QUERY)" --k $(K)

search_dataset:
	uv run python3 -m src search_dataset

install:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
