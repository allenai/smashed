"""
Unit test for printing mappers

Author: Luca Soldaini
Email:  lucas@allenai.org
"""

import unittest

from smashed.mappers.debug import MockMapper


class TestPrint(unittest.TestCase):
    """Test if pipelines compose correctly"""

    def test_mapper_repr(self):
        """Test if mappers can be detached from a pipeline"""
        mapper = MockMapper(1)

        self.assertEqual(str(mapper), f"MockMapper({mapper.fingerprint})")

    def test_pipeline_repr(self):
        p = MockMapper(1) >> MockMapper(2)
        self.assertEqual(
            str(p),
            (
                f"MockMapper({p.fingerprint}) >> "
                f"MockMapper({p.pipeline.fingerprint})"  # pyright: ignore
            ),
        )
