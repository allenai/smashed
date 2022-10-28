import unittest

from smashed.mappers.nested import TextTruncateMapper, WordsTruncateMapper
from smashed.mappers.promptsource import (
    DatasetPromptsourceMapper,
    JinjaPromptsourceMapper,
    PromptsourceMapper,
)


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

    def test_text_truncate_mapper(self):
        data = [
            {
                "question": "What is the capital of France? " * 50,
                "context": "Paris is the capital of France. " * 50,
                "answers": {
                    "text": ["Paris " * 50, "Paris " * 50],
                    "answer_start": [0, 0],
                },
            },
        ]
        mapper = TextTruncateMapper(
            nested_fields={
                "question": 30,
                "context": 31,
                "answers.text.[]": 5,
            }
        )
        mapped_data = mapper.map(data, remove_columns=True)
        self.assertEqual(
            mapped_data[0]["question"], "What is the capital of France?"
        )
        self.assertEqual(
            mapped_data[0]["context"],
            "Paris is the capital of France.",
        )
        self.assertEqual(
            mapped_data[0]["answers"]["text"],
            ["Paris", "Paris"],
        )

        mapper = WordsTruncateMapper(
            nested_fields={
                "question": 6,
                "context": 6,
                "answers.text.[]": 1,
            },
            splitter="blingfire",
        )
        mapped_data = mapper.map(data, remove_columns=True)
        self.assertEqual(
            mapped_data[0]["question"], "What is the capital of France"
        )
        self.assertEqual(
            mapped_data[0]["context"], "Paris is the capital of France"
        )
        self.assertEqual(mapped_data[0]["answers"]["text"], ["Paris", "Paris"])
