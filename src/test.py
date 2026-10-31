from pathlib import Path
from utils import StudentSearchResultsAndAnswer

path = Path(
    "/goinfre/mohhnine/rag_git/data/output/answers/dataset_docs_public.json/dataset_code_public.json"
)

result = StudentSearchResultsAndAnswer.model_validate_json(
    path.read_text(encoding="utf-8")
)

print("Questions:", len(result.search_results))
print("k:", result.k)
print("Answer:", result.search_results[0].answer)

assert all(item.answer.strip() for item in result.search_results)