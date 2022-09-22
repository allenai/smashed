"""
Unit test for caching and loading from cache

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import tempfile
import unittest

from necessary import necessary

from smashed.base.mappers import SingleBaseMapper
from smashed.mappers import EndCachingMapper, StartCachingMapper

with necessary("datasets"):
    from datasets.arrow_dataset import Dataset


class MockMapper(SingleBaseMapper):
    """A mock mapper that returns the same data it receives.
    Used for testing."""

    def __init__(self):
        super().__init__()

    def transform(self, data: dict) -> dict:
        return {k: v + 1 for k, v in data.items()}


class TestCaching(unittest.TestCase):
    def test_list_cache(self):
        """Test if caching works"""

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = (
                StartCachingMapper(tmpdir)
                >> MockMapper()
                >> MockMapper()
                >> EndCachingMapper()
            )

            data = [{"a": 1, "b": 2}]
            out1 = pipeline.map(data)
            out2 = pipeline.map(data)

            self.assertEqual(out1, out2)

    def test_datasets_cache(self):
        dt = Dataset.from_dict(
            {"a": [i for i in range(5)], "b": [i**2 for i in range(5)]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = (
                StartCachingMapper(tmpdir)
                >> MockMapper()
                >> MockMapper()
                >> EndCachingMapper()
            )

            out1 = pipeline.map(dt)
            out2 = pipeline.map(dt)

            self.assertEqual([e for e in out1], [e for e in out2])
