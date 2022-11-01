import unittest
from tempfile import NamedTemporaryFile

from transformers.models.bert import BertTokenizerFast

from smashed.mappers.prompting import (
    EncodeFieldsMapper,
    FillEncodedPromptMapper,
    TruncateMultipleFieldsMapper,
)
from smashed.recipes.prompting import PromptingRecipe


class TestTruncate(unittest.TestCase):
    def test_lengths_algo_uniform(self):
        lens = [10, 8, 6, 4, 2]
        max_len = 20
        truncated = [6, 5, 4, 2, 1]
        self.assertEqual(
            TruncateMultipleFieldsMapper._find_truncated_lens_uniform(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

        lens = [50, 40, 1, 1, 1]
        max_len = 20
        truncated = [10, 8, 0, 0, 0]
        self.assertEqual(
            TruncateMultipleFieldsMapper._find_truncated_lens_uniform(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

    def test_lengths_algo_longest(self):
        lens = [10, 8, 6, 4, 2]
        max_len = 20
        truncated = [5, 4, 4, 4, 2]
        self.assertEqual(
            TruncateMultipleFieldsMapper._find_truncated_lens_longest(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

        lens = [50, 40, 1, 1, 1]
        max_len = 20
        truncated = [9, 7, 1, 1, 1]
        self.assertEqual(
            TruncateMultipleFieldsMapper._find_truncated_lens_longest(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

    def _make_tokenizer(self) -> BertTokenizerFast:
        with NamedTemporaryFile(mode="r+") as f:
            vocab = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "hello",
                "world",
                "this",
                "is",
                "a",
                "test",
                "hi",
                "there",
                "many",
                "##i",
                "with",
                "the",
                "of",
            ]
            f.write("\n".join(vocab))
            f.flush()
            tokenizer = BertTokenizerFast(f.name, do_lower_case=True)

        tokenizer.model_max_length = 32
        return tokenizer

    def test_encode_offset(self):
        tokenizer = self._make_tokenizer()

        mapper = EncodeFieldsMapper(
            fields_to_encode=["a", "b", "c"],
            tokenizer=tokenizer,
            fields_to_return_offset_mapping=("a",),
        )
        data = [
            {
                "a": "many  hello world",
                "b": "hiii there",
                "c": "this is a test",
            }
        ]

        data = mapper.map(data)

        self.assertIn("offset_a", data[0])
        self.assertEqual(
            data[0]["offset_a"],
            [[0, 4], [6, 11], [12, 17]],
        )

    def test_encode(self):
        tokenizer = self._make_tokenizer()

        mapper = EncodeFieldsMapper(
            fields_to_encode=["a", "b", "c"],
            tokenizer=tokenizer,
        ) >> TruncateMultipleFieldsMapper(
            fields_to_truncate=["a", "b"],
            fields_to_preserve=["c"],
            max_length=16,
            strategy="longest",
        )
        data = [
            {
                "a": "many " * 30 + " hello world",
                "b": "hi" + "i" * 10 + " there",
                "c": "this is a test",
            }
        ]

        predicted = mapper.map(data)
        reference = [
            {
                "a": [12, 12, 12, 12, 12, 12],
                "b": [10, 13, 13, 13, 13, 13],
                "c": [6, 7, 8, 9],
            }
        ]
        self.assertEqual(predicted, reference)

        mapper.pipeline.strategy = "uniform"  # pyright: ignore
        predicted = mapper.map(data)
        reference = [
            {
                "a": [12, 12, 12, 12, 12, 12, 12, 12],
                "b": [10, 13, 13],
                "c": [6, 7, 8, 9],
            }
        ]
        self.assertEqual(predicted, reference)

    def test_fill(self):
        tokenizer = self._make_tokenizer()
        mapper = (
            EncodeFieldsMapper(
                fields_to_encode=["a", "b", "c"],
                tokenizer=tokenizer,
            )
            >> TruncateMultipleFieldsMapper(
                fields_to_truncate=["a", "b"],
                fields_to_preserve=["c"],
                max_length=16,
                strategy="uniform",
            )
            >> FillEncodedPromptMapper(
                template="{a} is a {b} with the help of {c}.",
                tokenizer=tokenizer,
            )
        )

        data = [
            {
                "a": "many " * 30 + " hello world",
                "b": "hi" + "i" * 10 + " there",
                "c": "this is a test",
            }
        ]

        output = mapper.map(data, remove_columns=True)

        reference = [
            {
                "input_ids": (
                    # {a}
                    [12, 12, 12, 12, 12, 12, 12, 12]
                    # is a
                    + [7, 8]
                    # {b}
                    + [10, 13, 13]
                    # with the help of
                    + [14, 15, 1, 16]
                    # {c}
                    + [6, 7, 8, 9]
                    # .
                    + [1]
                ),
                "attention_mask": [1] * 22,
            }
        ]

        self.assertEqual(len(output), len(reference))
        self.assertEqual(output[0].keys(), reference[0].keys())
        self.assertEqual(output[0]["input_ids"], reference[0]["input_ids"])
        self.assertEqual(
            output[0]["attention_mask"], reference[0]["attention_mask"]
        )

    def test_recipe(self):
        tokenizer = self._make_tokenizer()
        recipe = PromptingRecipe(
            tokenizer=tokenizer,
            source_template="{a} is a {b} with the help of {c}.",
            max_source_length=22,
            fields_to_truncate=["a", "b"],
            strategy="uniform",
        )

        data = [
            {
                "a": "many " * 30 + " hello world",
                "b": "hi" + "i" * 10 + " there",
                "c": "this is a test",
            }
        ]

        output = recipe.map(data)

        reference = [
            {
                "input_ids": (
                    # {a}
                    [12, 12, 12, 12, 12, 12, 12, 12]
                    # is a
                    + [7, 8]
                    # {b}
                    + [10, 13, 13]
                    # with the help of
                    + [14, 15, 1, 16]
                    # {c}
                    + [6, 7, 8, 9]
                    # .
                    + [1]
                ),
                "attention_mask": [1] * 22,
            }
        ]

        self.assertEqual(len(output), len(reference))
        self.assertEqual(output[0].keys(), reference[0].keys())
        self.assertEqual(output[0]["input_ids"], reference[0]["input_ids"])
        self.assertEqual(
            output[0]["attention_mask"], reference[0]["attention_mask"]
        )
