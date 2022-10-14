import unittest
from copy import deepcopy
from functools import partial

from datasets.arrow_dataset import Batch, Dataset

from smashed.mappers.debug import MockMapper


class TestBatchInterface(unittest.TestCase):
    def test_batch(self, remove_columns: bool = False):
        mapper = MockMapper(1, output_fields=["a"])

        data = Dataset.from_list([{"a": i, "b": i**2} for i in range(100)])

        def _batch_fn(data: Batch, mapper: MockMapper) -> Batch:
            return mapper.map(deepcopy(data), remove_columns=remove_columns)

        fn = partial(_batch_fn, mapper=mapper)

        out = data.map(
            fn,
            batched=True,
            batch_size=10,
            remove_columns=["b"] if remove_columns else None,
        )

        self.assertEqual(len(out), 100)
        if remove_columns:
            self.assertEqual([f for f in out.features.keys()], ["a"])
        else:
            self.assertEqual([f for f in out.features.keys()], ["a", "b"])

        self.assertEqual(out["a"], [i + 1 for i in range(100)])

    def test_batch_remove_columns(self):
        self.test_batch(remove_columns=True)
