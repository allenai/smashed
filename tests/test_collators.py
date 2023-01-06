"""
Unit test for collating sequences of uneven length

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import unittest

from transformers.models.auto.tokenization_auto import AutoTokenizer

from smashed.mappers.batchers import FixedBatchSizeMapper
from smashed.mappers.collators import (
    FromTokenizerTensorCollatorMapper,
    ListCollatorMapper,
    TensorCollatorMapper,
)
from smashed.mappers.converters import Python2TorchMapper


class TestListCollators(unittest.TestCase):
    def test_base_collator(self):
        dataset = [
            {"a": [1, 2, 3], "b": [11, 12]},
            {"a": [4, 5], "b": [13]},
            {"a": [6, 7, 8, 9, 10], "b": [14]},
            {"a": [15], "b": [15, 16, 17, 18, 19, 20]},
            {"a": [21, 22], "b": [23, 24, 25]},
        ]
        pipeline = FixedBatchSizeMapper(batch_size=3) >> ListCollatorMapper(
            fields_pad_ids={"a": -1, "b": -2}
        )

        collated_dataset = pipeline.map(dataset)

        self.assertEqual(len(collated_dataset), 2)

        for field in ("a", "b"):
            self.assertEqual(len(collated_dataset[0][field]), 3)
            self.assertEqual(len(collated_dataset[1][field]), 2)

        self.assertEqual(
            collated_dataset[0]["a"],
            [[1, 2, 3, -1, -1], [4, 5, -1, -1, -1], [6, 7, 8, 9, 10]],
        )
        self.assertEqual(
            collated_dataset[0]["b"], [[11, 12], [13, -2], [14, -2]]
        )
        self.assertEqual(
            collated_dataset[1]["a"],
            [
                [15, -1],
                [21, 22],
            ],
        )
        self.assertEqual(
            collated_dataset[1]["b"],
            [[15, 16, 17, 18, 19, 20], [23, 24, 25, -2, -2, -2]],
        )

    def test_nested_collators(self):
        dataset = [
            {"a": [[1.0, 1.1], [2.0], [3.0, 3.1, 3.2, 3.3]], "b": [11, 12]},
            {"a": [[4.0, 4.1, 4.2, 4.3, 4.4], [5.0, 5.1]], "b": [13]},
        ]

        pipeline = FixedBatchSizeMapper(batch_size=2) >> ListCollatorMapper(
            fields_pad_ids={"a": -1, "b": -2}
        )

        collated_dataset = pipeline.map(dataset)

        self.assertTrue(len(collated_dataset) == 1)
        for seq in collated_dataset[0]["a"]:
            self.assertEqual(len(seq), 3)
            for inner_seq in seq:
                self.assertEqual(len(inner_seq), 5)

        grouped_a = collated_dataset[0]["a"]
        self.assertEqual(grouped_a[0][0], [1.0, 1.1, -1, -1, -1])
        self.assertEqual(grouped_a[0][1], [2.0, -1, -1, -1, -1])
        self.assertEqual(grouped_a[0][2], [3.0, 3.1, 3.2, 3.3, -1])
        self.assertEqual(grouped_a[1][0], [4.0, 4.1, 4.2, 4.3, 4.4])
        self.assertEqual(grouped_a[1][1], [5.0, 5.1, -1, -1, -1])
        self.assertEqual(grouped_a[1][2], [-1, -1, -1, -1, -1])

    def test_left_padding(self):
        dataset = [
            {"a": [1, 2, 3]},
            {"a": [4, 5]},
            {"a": [6, 7, 8, 9, 10]},
        ]
        pipeline = FixedBatchSizeMapper(
            batch_size="max"
        ) >> ListCollatorMapper(
            fields_pad_ids={"a": -1}, left_pad_fields=["a"]
        )

        output = pipeline.map(dataset)

        self.assertEqual(len(output[0]["a"]), 3)
        self.assertEqual([len(s) for s in output[0]["a"]], [5, 5, 5])
        self.assertEqual(output[0]["a"][0], [-1, -1, 1, 2, 3])
        self.assertEqual(output[0]["a"][1], [-1, -1, -1, 4, 5])
        self.assertEqual(output[0]["a"][2], [6, 7, 8, 9, 10])


class TestTensorCollators(unittest.TestCase):
    def test_base_collator(self):
        dataset = [
            {"a": [1, 2, 3], "b": [11, 12]},
            {"a": [4, 5], "b": [13]},
            {"a": [6, 7, 8, 9, 10], "b": [14]},
        ]
        pipeline = (
            Python2TorchMapper()
            >> FixedBatchSizeMapper(batch_size=3)
            >> TensorCollatorMapper(fields_pad_ids={"a": -1, "b": -2})
        )

        collated_dataset = pipeline.map(dataset)
        self.assertEqual(len(collated_dataset), 1)
        self.assertEqual(collated_dataset[0]["a"].shape, (3, 5))
        self.assertEqual(collated_dataset[0]["b"].shape, (3, 2))
        self.assertEqual((collated_dataset[0]["a"] == -1).sum(), 5)
        self.assertEqual((collated_dataset[0]["b"] == -2).sum(), 2)

    def test_from_tokenizer_collator(self):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        sent_first = "I am a sentence"  # len 4 when tokenized
        sent_last = "I am a pterodactyl"  # len 8 when tokenized

        dataset = [tokenizer(sent_first), tokenizer(sent_last)]

        pipeline = (
            Python2TorchMapper()
            >> FixedBatchSizeMapper(batch_size=2)
            >> FromTokenizerTensorCollatorMapper(tokenizer)
        )
        collated_dataset = pipeline.map(dataset)

        self.assertEqual(len(collated_dataset), 1)

        # check if we padded to the length of the 2nd sentence
        # (which is the longest at 8 subwords) + 2 (for CLS and SEP)
        self.assertEqual(collated_dataset[0]["input_ids"].shape, (2, 8 + 2))
        # the first sequence is 4 subwords long, while the second is 8;
        # therefore, there should be 4 padding tokens in the collated batch
        self.assertEqual(
            (collated_dataset[0]["input_ids"] == tokenizer.pad_token_id).sum(),
            4,
        )

        # same thing except attention mask uses 0 for padding
        self.assertEqual((collated_dataset[0]["attention_mask"] == 0).sum(), 4)

    def test_left_padding(self):
        dataset = [
            {"a": [1, 2, 3]},
            {"a": [4, 5]},
            {"a": [6, 7, 8, 9, 10]},
        ]
        pipeline = (
            Python2TorchMapper()
            >> FixedBatchSizeMapper(batch_size="max")
            >> TensorCollatorMapper(
                fields_pad_ids={"a": -1}, left_pad_fields=["a"]
            )
        )

        output = pipeline.map(dataset)

        self.assertEqual(output[0]["a"].shape, (3, 5))
        self.assertEqual(output[0]["a"][0].tolist(), [-1, -1, 1, 2, 3])
        self.assertEqual(output[0]["a"][1].tolist(), [-1, -1, -1, 4, 5])
        self.assertEqual(output[0]["a"][2].tolist(), [6, 7, 8, 9, 10])
