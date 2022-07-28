"""

Tests for ShapeMappers module

@kylel

"""

import unittest

from smashed.interfaces.simple import Dataset, FlattenMapper


class TestFlattenMapper(unittest.TestCase):
    """Test FlattenMapper"""
    def test_map(self):
        mapper = FlattenMapper(field='input_ids')
        dataset = Dataset([
            {
                'input_ids': [[1, 2, 3, 4], [5, 6, 7, 8]]
            },
            {
                'input_ids': [[9, 10, 11, 12], [13, 14, 15, 16]]
            }
        ])
        new_dataset = mapper.map(dataset)
        assert new_dataset == [
            {'input_ids': [1, 2, 3, 4, 5, 6, 7, 8]},
            {'input_ids': [9, 10, 11, 12, 13, 14, 15, 16]}
        ]
