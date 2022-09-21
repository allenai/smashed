"""
Unit test for composing mappers and pipelines

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import unittest

from smashed.base.mappers import SingleBaseMapper

# from smashed.base.pipeline import Pipeline


class MockMapper(SingleBaseMapper):
    """A mock mapper that returns the same data it receives.
    Used for testing."""

    __slots__ = ("stage",)

    def __init__(self, stage: int):
        super().__init__()
        self.stage = stage

    def transform(self, data: dict) -> dict:
        return {"stage": (data.get("stage", []) + [self.stage])}


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
        self.assertEqual(pipeline.stage, 1)

        self.assertIsInstance(pipeline.pipeline, MockMapper)
        self.assertEqual(pipeline.pipeline.stage, 2)  # pyright: ignore

        self.assertIsInstance(
            pipeline.pipeline.pipeline, MockMapper  # pyright: ignore
        )
        self.assertEqual(
            pipeline.pipeline.pipeline.stage, 3  # pyright: ignore
        )

        self.assertEqual(
            pipeline.pipeline.pipeline.pipeline, None  # pyright: ignore
        )

    def test_run_pipeline(self):
        """Test a full pipeline"""
        pipeline = MockMapper(1) >> MockMapper(2) >> MockMapper(3)

        dataset = [{"stage": [0]}]
        dataset = pipeline.map(dataset)

        self.assertEqual(dataset[0]["stage"], [0, 1, 2, 3])
