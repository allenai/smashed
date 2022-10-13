import unittest

from smashed.mappers.prompting import TruncateFieldsMapper


class TestTruncate(unittest.TestCase):
    def test_lengths_algo_uniform(self):
        lens = [10, 8, 6, 4, 2]
        max_len = 20
        truncated = [6, 5, 4, 2, 1]
        self.assertEqual(
            TruncateFieldsMapper._find_truncated_lens_uniform(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

        lens = [50, 40, 1, 1, 1]
        max_len = 20
        truncated = [10, 8, 0, 0, 0]
        self.assertEqual(
            TruncateFieldsMapper._find_truncated_lens_uniform(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

    def test_lengths_algo_longest(self):
        lens = [10, 8, 6, 4, 2]
        max_len = 20
        truncated = [5, 4, 4, 4, 2]
        self.assertEqual(
            TruncateFieldsMapper._find_truncated_lens_longest(
                lens=lens, max_len=max_len
            ),
            truncated,
        )

        lens = [50, 40, 1, 1, 1]
        max_len = 20
        truncated = [9, 7, 1, 1, 1]
        self.assertEqual(
            TruncateFieldsMapper._find_truncated_lens_longest(
                lens=lens, max_len=max_len
            ),
            truncated,
        )
