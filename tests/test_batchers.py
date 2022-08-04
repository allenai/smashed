"""
Unit test for making batches from a dataset

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import unittest

from smashed.mappers.batchers import FixedBatchSizeMapper


class TestBatchers(unittest.TestCase):
    def test_batchers(self):
        dataset = [
            {k: [i, v] for v, k in enumerate("a b c".split())}
            for i in range(10)
        ]
        mapper = FixedBatchSizeMapper(batch_size=3)
        batched_dataset = mapper.map(dataset)

        self.assertEqual(len(batched_dataset), 4)

        for key in "a b c".split():
            self.assertEqual(len(batched_dataset[0][key]), 3)
            self.assertEqual(len(batched_dataset[1][key]), 3)
            self.assertEqual(len(batched_dataset[2][key]), 3)
            self.assertEqual(len(batched_dataset[3][key]), 1)

        self.assertEqual(
            batched_dataset[0],
            {
                "a": [[0, 0], [1, 0], [2, 0]],
                "b": [[0, 1], [1, 1], [2, 1]],
                "c": [[0, 2], [1, 2], [2, 2]],
            },
        )
