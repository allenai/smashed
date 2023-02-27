import unittest

from smashed.mappers.glom import GlomMapper
from smashed.mappers.text import (
    FtfyMapper,
    TextToWordsMapper,
    WordsToTextMapper,
)
from smashed.mappers.tokenize import TruncateSingleFieldMapper


class TestFtfyMapper(unittest.TestCase):
    def test_ftfy_mapper(self):
        # These are the test cases from the ftfy documentation
        dataset = [
            {"text": "âœ” No problems"},
            {"text": "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."},
            {"text": "l’humanitÃ©"},
            {"text": "Ã\xa0 perturber la rÃ©flexion"},
            {"text": "Ã perturber la rÃ©flexion"},
            {"text": "P&EACUTE;REZ"},
        ]

        mapper = FtfyMapper(input_fields="text")

        result = mapper.map(dataset)

        self.assertEqual(result[0], {"text": "✔ No problems"})
        self.assertEqual(
            result[1], {"text": "The Mona Lisa doesn't have eyebrows."}
        )
        self.assertEqual(result[2], {"text": "l'humanité"})
        self.assertEqual(result[3], {"text": "à perturber la réflexion"})
        self.assertEqual(result[4], {"text": "à perturber la réflexion"})
        self.assertEqual(result[5], {"text": "PÉREZ"})


class TestToWords(unittest.TestCase):
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
        mapper = GlomMapper(
            spec_fields={
                "question": "question",
                "context": ("context",),
                "answer": ("answers", "text", "-1"),
            }
        ) >> TruncateSingleFieldMapper(
            fields_to_truncate={
                "question": 30,
                "context": 31,
                "answer": 5,
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
        self.assertEqual(mapped_data[0]["answer"], "Paris")

    def test_word_truncate_mapper(self):
        data = [
            {
                "question": "What is the capital of France? " * 50,
                "context": "Paris is the capital of France. " * 50,
                "answer": "Paris " * 50,
            },
        ]

        mapper = (
            TextToWordsMapper(
                fields=["question", "context", "answer"],
                splitter="blingfire",
            )
            >> TruncateSingleFieldMapper(
                fields_to_truncate={
                    "question": 6,
                    "context": 6,
                    "answer": 1,
                }
            )
            >> WordsToTextMapper(
                fields=["question", "context", "answer"],
                joiner=" ",
            )
        )
        mapped_data = mapper.map(data, remove_columns=True)
        self.assertEqual(
            mapped_data[0]["question"], "What is the capital of France"
        )
        self.assertEqual(
            mapped_data[0]["context"], "Paris is the capital of France"
        )
        self.assertEqual(mapped_data[0]["answer"], "Paris")
