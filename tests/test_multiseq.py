import copy
import unittest

from smashed.interfaces.simple import Dataset
from smashed.mappers.multiseq import (
    AddPrefixSuffixMapper,
    AttentionMaskAddPrefixSuffixMapper,
    TokensSequencesAddPrefixSuffixMapper,
    TokenTypeIdsAddPrefixSuffixMapper,
)


class TestAddPrefixSuffixMapper(unittest.TestCase):
    def test_map(self):
        mapper = AddPrefixSuffixMapper(
            input_field="a",
            prefix="Translate into Ukrainian: ",
            suffix=" Translation: ",
        )
        dataset = Dataset(
            [
                {"a": ["I am the first!", "No, I am the first!"]},
                {"a": ["I am the second!", "No, I am the second!"]},
            ]
        )
        mapped_dataset = mapper.map(copy.deepcopy(dataset))

        self.assertEqual(len(dataset), 2)
        for i in range(len(dataset)):
            pair = dataset[i]["a"]
            expected = {
                "a": [
                    f"Translate into Ukrainian: {pair[0]} Translation: ",
                    f"Translate into Ukrainian: {pair[1]} Translation: ",
                ]
            }
            self.assertEqual(mapped_dataset[i], expected)


class TestTokensSequencesAddPrefixSuffixMapper(unittest.TestCase):
    def test_map(self):
        mapper = TokensSequencesAddPrefixSuffixMapper(
            prefix=[123456],
            suffix=[123, 456],
        )
        dataset = Dataset(
            [
                {"input_ids": [[7, 7, 7], [7, 7]]},
                {"input_ids": [[9, 9, 9, 9], [9, 9, 9]]},
            ]
        )
        mapped_dataset = mapper.map(copy.deepcopy(dataset))

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            mapped_dataset[0]["input_ids"],
            [[123456, 7, 7, 7, 123, 456], [123456, 7, 7, 123, 456]],
        )
        self.assertEqual(
            mapped_dataset[1]["input_ids"],
            [[123456, 9, 9, 9, 9, 123, 456], [123456, 9, 9, 9, 123, 456]],
        )


class TestAttentionMaskAddPrefixSuffixMapper(unittest.TestCase):
    def test_map(self):
        num_prefix_tokens = 3
        mapper = AttentionMaskAddPrefixSuffixMapper(
            num_prefix_tokens=num_prefix_tokens,
        )
        dataset = Dataset(
            [
                {"attention_mask": [[1, 1, 1], [1, 1, 0]]},
                {"attention_mask": [[1, 1, 1, 1], [1, 0, 0, 0]]},
            ]
        )
        mapped_dataset = mapper.map(copy.deepcopy(dataset))

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            mapped_dataset[0]["attention_mask"],
            [
                num_prefix_tokens * [1] + [1, 1, 1],
                num_prefix_tokens * [1] + [1, 1, 0],
            ],
        )
        self.assertEqual(
            mapped_dataset[1]["attention_mask"],
            [
                num_prefix_tokens * [1] + [1, 1, 1, 1],
                num_prefix_tokens * [1] + [1, 0, 0, 0],
            ],
        )


class TestTokenTypeIdsAddPrefixSuffixMapper(unittest.TestCase):
    def test_map(self):
        num_prefix_tokens = 3
        mapper = TokenTypeIdsAddPrefixSuffixMapper(
            num_prefix_tokens=num_prefix_tokens,
        )
        dataset = Dataset(
            [
                {"token_type_ids": [[0, 0, 0], [1, 1, 1]]},
                {"token_type_ids": [[0, 0, 0, 0], [1, 1, 1, 1]]},
            ]
        )
        mapped_dataset = mapper.map(copy.deepcopy(dataset))

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            mapped_dataset[0]["token_type_ids"],
            [
                num_prefix_tokens * [0] + [0, 0, 0],
                num_prefix_tokens * [1] + [1, 1, 1],
            ],
        )
        self.assertEqual(
            mapped_dataset[1]["token_type_ids"],
            [
                num_prefix_tokens * [0] + [0, 0, 0, 0],
                num_prefix_tokens * [1] + [1, 1, 1, 1],
            ],
        )
