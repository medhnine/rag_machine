import json
from pathlib import Path
from pydantic import ValidationError
from src.utils import AnsweredQuestion

path = Path("data/datasets/AnsweredQuestions/dataset_code_public.json")

data = json.loads(path.read_text(encoding="utf-8"))

first_question = data["rag_questions"][0]

try:
    question = AnsweredQuestion.model_validate(first_question)
    print("Valid AnsweredQuestion")
except ValidationError as e:
    print(e)