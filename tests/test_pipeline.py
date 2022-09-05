"""
Unit test for composing mappers and pipelines

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import unittest

from smashed.base.mappers import SingleBaseMapper
from smashed.base.pipeline import Pipeline


class MockMapper(SingleBaseMapper):
    """A mock mapper that returns the same data it receives.
    Used for testing."""

    def __init__(self, stage: int):
        super().__init__()
        self.stage = stage

    def transform(self, data: dict) -> dict:
        return {"stage": (data.get("stage", []) + [self.stage])}

    def __eq__(self, __o: object) -> bool:
        """Check if two mappers are equal; useful to
        check if pipelines are equal."""

        return isinstance(__o, MockMapper) and self.stage == __o.stage


class TestPipeline(unittest.TestCase):
    """Test if pipelines compose correctly"""

    def test_rshift_lshift_implementations(self):
        """Test if two mappers compose correctly"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        pipeline = Pipeline(mapper1) >> Pipeline(mapper2)
        self.assertEqual(pipeline, Pipeline(mapper1, mapper2))

    def test_mappers_to_pipeline(self):
        """Test if mappers can be used as pipelines"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        pipeline = mapper1 >> mapper2
        self.assertEqual(pipeline, Pipeline(mapper1, mapper2))

    def test_multiple_mappers_to_pipeline(self):
        """Test if multiple mappers can be used as pipelines"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        mapper3 = MockMapper(3)
        pipeline = mapper1 >> mapper2 >> mapper3
        self.assertEqual(pipeline, Pipeline(mapper1, mapper2, mapper3))

    def test_pipeline_order(self):
        """Test if the order of the mappers in a pipeline is respected"""
        mapper1 = MockMapper(1)
        mapper2 = MockMapper(2)
        pipeline1 = mapper1 >> mapper2
        pipeline2 = mapper1 << mapper2
        self.assertNotEqual(pipeline1, pipeline2)
        self.assertEqual(pipeline1.mappers, pipeline2.mappers[::-1])

    def test_run_pipeline(self):
        """Test a full pipeline"""
        pipeline = MockMapper(1) >> MockMapper(2) >> MockMapper(3)

        dataset = [{"stage": [0]}]
        dataset = pipeline.map(dataset)

        self.assertEqual(dataset[0]["stage"], [0, 1, 2, 3])
