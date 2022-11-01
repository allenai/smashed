"""

Tests for ShapeMappers module

@kylel

"""

import unittest

from smashed.mappers.shape import FlattenMapper, SingleSequenceStriderMapper


class TestFlattenMapper(unittest.TestCase):
    """Test FlattenMapper"""

    def test_map(self):
        mapper = FlattenMapper(field="input_ids")
        dataset = [
            {"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]},
            {"input_ids": [[9, 10, 11, 12], [13, 14, 15, 16]]},
        ]
        new_dataset = mapper.map(dataset)
        assert new_dataset == [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"input_ids": [9, 10, 11, 12, 13, 14, 15, 16]},
        ]

    def test_stride(self):
        mapper = SingleSequenceStriderMapper(
            field_to_stride="input_ids", max_length=3, stride=1
        )
        dataset = [
            {"input_ids": [1, 2, 3, 4]},
            {"input_ids": [5, 6, 7, 8]},
        ]
        new_dataset = mapper.map(dataset)

        self.assertEqual(len(new_dataset), 4)
        self.assertEqual(
            new_dataset,
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [2, 3, 4]},
                {"input_ids": [5, 6, 7]},
                {"input_ids": [6, 7, 8]},
            ],
        )

        mapper = SingleSequenceStriderMapper(
            field_to_stride="input_ids", max_length=2, stride=2
        )
        dataset = [
            {"input_ids": [1, 2, 3, 4]},
            {"input_ids": [5, 6, 7, 8]},
        ]
        new_dataset = mapper.map(dataset)
        self.assertEqual(len(new_dataset), 4)
        self.assertEqual(
            new_dataset,
            [
                {"input_ids": [1, 2]},
                {"input_ids": [3, 4]},
                {"input_ids": [5, 6]},
                {"input_ids": [7, 8]},
            ],
        )

        mapper = SingleSequenceStriderMapper(
            field_to_stride="input_ids", max_length=3, stride=3, keep_last=True
        )
        new_dataset = mapper.map(dataset)
        self.assertEqual(len(new_dataset), 4)
        self.assertEqual(
            new_dataset,
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4]},
                {"input_ids": [5, 6, 7]},
                {"input_ids": [8]},
            ],
        )
