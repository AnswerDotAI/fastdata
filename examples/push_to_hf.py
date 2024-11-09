from fastcore.utils import *
from fastdata.core import FastData


class Translation:
    "Translation from an English phrase to a Spanish phrase"

    def __init__(self, english: str, spanish: str):
        self.english = english
        self.spanish = spanish

    def __repr__(self):
        return f"{self.english} ➡ *{self.spanish}*"


prompt_template = """\
Generate English and Spanish translations on the following topic:
<topic>{topic}</topic>
"""

inputs = [
    {"topic": "I am going to the beach this weekend"},
    {"topic": "I am going to the gym after work"},
    {"topic": "I am going to the park with my kids"},
    {"topic": "I am going to the movies with my friends"},
    {"topic": "I am going to the store to buy some groceries"},
    {"topic": "I am going to the library to read some books"},
    {"topic": "I am going to the zoo to see the animals"},
    {"topic": "I am going to the museum to see the art"},
    {"topic": "I am going to the restaurant to eat some food"},
]

fast_data = FastData(model="claude-3-haiku-20240307")
dataset_name = "my_dataset"

repo_id, translations = fast_data.generate_to_hf(
    prompt_template=prompt_template,
    inputs=inputs,
    schema=Translation,
    repo_id=dataset_name,
    save_interval=4,
)
print(f"A new repository has been create on {repo_id}")
print(translations)
