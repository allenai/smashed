import unittest

from transformers.models.auto import AutoTokenizer

from smashed.mappers.promptsource import (
    DatasetPromptsourceMapper,
    JinjaPromptsourceMapper,
    PromptsourceMapper,
)
from smashed.recipes.promptsource import PromptsourceRecipe


class TestPromptsource(unittest.TestCase):
    def test_jinja_prompt_source_mapper(self):
        mapper = JinjaPromptsourceMapper(
            jinja="Q: {{question}}\nA: |||{{answers.text[0]}}"
        )
        dataset = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "answers": {"text": ["Paris"], "answer_start": [0]},
            }
        ]
        mapped_dataset = mapper.map(dataset, remove_columns=True)
        self.assertEqual(
            mapped_dataset[0]["source"],
            "Q: What is the capital of France?\nA:",
        )
        self.assertEqual(mapped_dataset[0]["target"], "Paris")

    def test_dataset_prompt_source_mapper(self):
        mapper = DatasetPromptsourceMapper(
            dataset_name="squad",
            template_name="given_context_answer_question_variation",
        )
        dataset = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
                "answers": {"text": ["Paris"], "answer_start": [0]},
            }
        ]

        mapped_dataset = mapper.map(dataset, remove_columns=True)

        self.assertEqual(len(mapped_dataset), 1)
        self.assertEqual(len(mapped_dataset[0]), 2)
        self.assertEqual(
            mapped_dataset[0]["source"],
            (
                "Paris is the capital of France.\n\n"
                "Q: What is the capital of France?\n\nA:"
            ),
        )
        self.assertEqual(mapped_dataset[0]["target"], "Paris")

        mapper2 = PromptsourceMapper(mapper.template)
        mapped_dataset2 = mapper2.map(dataset, remove_columns=True)
        self.assertEqual(mapped_dataset, mapped_dataset2)

    def test_promptsource_recipe(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        recipe = PromptsourceRecipe(
            tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"),
            jinja_template="Q: {{question}}\nC: {{context}}\nA: |||{{answer}}",
            max_source_content_length=15,
            max_target_content_length=5,
        )
        dataset = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of " + ("France " * 10),
                "answer": "Paris " * 10,
            }
        ]

        mapped_dataset, *_ = recipe.map(dataset)

        self.assertEqual(
            tokenizer.decode(mapped_dataset["input_ids"]),
            (
                "Q : What is the capital of France? "
                "C : Paris is the capital of France "
                "A :"
            ),
        )

        self.assertEqual(
            tokenizer.decode(mapped_dataset["labels"]),
            "Paris Paris Paris Paris Paris",
        )
