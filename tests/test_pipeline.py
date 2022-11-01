"""
Unit test for composing mappers and pipelines

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import copy
import unittest

from smashed.base import make_pipeline
from smashed.mappers.debug import MockMapper


class TestPipeline(unittest.TestCase):
    """Test if pipelines compose correctly"""

    def test_equality(self):
        """Test if mappers can be detached from a pipeline"""
        mapper1 = MockMapper(1)
        pipeline = mapper1 >> MockMapper(2)

        self.assertNotEqual(mapper1, pipeline)
        self.assertEqual(mapper1, pipeline.detach())

    def test_rshift_lshift_implementations(self):
        """Test if two mappers compose correctly"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        pipeline = mapper1 >> mapper2
        self.assertEqual(pipeline.detach(), mapper1)
        self.assertEqual(
            pipeline.pipeline.detach(), mapper2  # pyright: ignore
        )

    def test_mappers_to_pipeline(self):
        """Test if mappers can be used as pipelines"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        pipeline = mapper2 << mapper1
        self.assertEqual(pipeline.detach(), mapper1)
        self.assertEqual(
            pipeline.pipeline.detach(), mapper2  # pyright: ignore
        )

    def test_multiple_mappers_to_pipeline(self):
        """Test if multiple mappers can be used as pipelines"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        mapper3 = MockMapper(3)
        pipeline = mapper1 >> mapper2 >> mapper3

        self.assertIsInstance(pipeline, MockMapper)
        self.assertEqual(pipeline.value, 1)

        self.assertIsInstance(pipeline.pipeline, MockMapper)
        self.assertEqual(pipeline.pipeline.value, 2)  # pyright: ignore

        self.assertIsInstance(
            pipeline.pipeline.pipeline, MockMapper  # pyright: ignore
        )
        self.assertEqual(
            pipeline.pipeline.pipeline.value, 3  # pyright: ignore
        )

        self.assertEqual(
            pipeline.pipeline.pipeline.pipeline, None  # pyright: ignore
        )

    def test_run_pipeline(self):
        """Test a full pipeline"""
        pipeline = MockMapper([1]) >> MockMapper([2]) >> MockMapper([3])

        dataset = [{"stage": [0]}]
        dataset = pipeline.map(dataset)

        self.assertEqual(dataset[0]["stage"], [0, 1, 2, 3])

    def test_make_pipeline_function(self):
        pipeline = make_pipeline(
            MockMapper([1]), MockMapper([2]), MockMapper([3])
        )

        dataset = [{"stage": [0]}]
        dataset = pipeline.map(dataset)

        self.assertEqual(dataset[0]["stage"], [0, 1, 2, 3])

    def test_reconstruction_pipeline(self):
        p1 = MockMapper([1]) >> MockMapper([2]) >> MockMapper([3])
        p2 = (
            p1.detach()
            >> p1.pipeline.detach()  # pyright: ignore
            >> p1.pipeline.pipeline.detach()  # pyright: ignore
        )
        p3 = copy.deepcopy(p1)
        self.assertEqual(p1, p2)
        self.assertEqual(p1, p3)

        dataset = [{"stage": [0]}]
        d1 = p1.map(dataset)
        d2 = p2.map(dataset)
        d3 = p3.map(dataset)
        self.assertEqual(d1, d2)
        self.assertEqual(d1, d3)
