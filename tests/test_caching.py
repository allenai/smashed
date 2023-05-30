"""
Unit test for caching and loading from cache

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import tempfile
import unittest

from necessary import necessary

from smashed.mappers import EndCachingMapper, StartCachingMapper
from smashed.mappers.debug import MockMapper

with necessary("datasets"):
    from datasets.arrow_dataset import Dataset


class TestCaching(unittest.TestCase):
    def test_list_cache(self):
        """Test if caching works"""

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = (
                StartCachingMapper(tmpdir)
                >> MockMapper(1)
                >> MockMapper(1)
                >> EndCachingMapper()
            )

            data = [{"a": 1, "b": 2}]
            out1 = pipeline.map(data)
            out2 = pipeline.map(data)

            self.assertEqual(out1, out2)

    def test_datasets_cache(self):
        dt = Dataset.from_dict(
            {"a": [i for i in range(5)], "b": [i ** 2 for i in range(5)]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = (
                StartCachingMapper(tmpdir)
                >> MockMapper(1)
                >> MockMapper(1)
                >> EndCachingMapper()
            )

            out1 = pipeline.map(dt)
            out2 = pipeline.map(dt)

            self.assertEqual([e for e in out1], [e for e in out2])

    def test_fails(self):
        """Test if caching fails when it should"""

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = (
                StartCachingMapper(tmpdir) >> MockMapper(1) >> MockMapper(1)
            )

            with self.assertRaises(ValueError):
                # This should fail because we didn't end the caching
                pipeline.map([{"a": 1, "b": 2}])

            pipeline = MockMapper(1) >> MockMapper(1) >> EndCachingMapper()

            with self.assertRaises(ValueError):
                # This should fail because we didn't start the caching
                pipeline.map([{"a": 1, "b": 2}])
