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
    TensorCollatorMapper,
)
from smashed.mappers.converters import Python2TorchMapper


class TestCollators(unittest.TestCase):
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
