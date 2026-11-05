import json
from pathlib import Path

source = Path(
    "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
)

target = Path("data/tests/search_results_docs_2.json")

data = json.loads(source.read_text(encoding="utf-8"))

data["search_results"] = data["search_results"][:2]

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data, indent=2), encoding="utf-8")