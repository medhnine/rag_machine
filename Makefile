SZIE=2000

run:
	uv run python3 -m src $(SZIE)

install:
	uv sync