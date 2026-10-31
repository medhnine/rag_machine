MAX_CHUNK_SIZE ?= 2000
QUERY ?= vllm uses
DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
K ?= 5

SAVE_DIRECTORY ?= data/output/search_results/UnansweredQuestions
STUDENT_SEARCH_RESULTS_PATH ?= data/output/search_results/UnansweredQuestions/dataset_docs_public.json
SAVE ?= data/output/search_results_and_answer/UnansweredQuestions


all:
	uv run python3 -m src

index:
	uv run python3 -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python3 -m src search --query "$(QUERY)" --k $(K)

search_dataset:
	uv run python3 -m src search_dataset --dataset_path $(DATASET_PATH) --k $(K) --save_directory $(SAVE_DIRECTORY) 

answer_dataset:
	uv run python3 -m src answer_dataset --student_search_results_path $(STUDENT_SEARCH_RESULTS_PATH) --save_directory $(SAVE)

answer:
	uv run python3 -m src answer --query "$(QUERY)" --k $(K)

install:
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
